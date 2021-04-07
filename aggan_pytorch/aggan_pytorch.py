import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random,randint,seed
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from kornia.filters import filter2D

import torchvision
from torchvision import transforms
from aggan_pytorch.version import __version__
from aggan_pytorch.diff_augment import DiffAugment

from pytorch_fid import fid_score

from vector_quantize_pytorch import VectorQuantize
from aggan_pytorch.ila import ImageLinearAttention
from aggan_pytorch.ban import BiAttention

from PIL import Image
from pathlib import Path

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png']
EPS = 1e-8
CALC_FID_NUM_IMAGES = 12800

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn, is_attn=False):
        super().__init__()
        self.is_attn = is_attn
        self.fn = fn
    def forward(self, x):
        if self.is_attn:
            out = self.fn(x)
            return(out[0] + x, *out[1:])
        else:
            return self.fn(x) + x

class Rezero(nn.Module):
    def __init__(self, fn, is_attn=False):   #only zero-start the 1st layer?
        super().__init__()
        self.is_attn = is_attn
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        if self.is_attn:
            # print("rezero shape:", x[0].shape, len(x))
            out = self.fn(x)
            return(out[0] * self.g, *out[1:])
        else:
            return self.fn(x) * self.g

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

# one layer of self-attention and feedforward, for images

class Attn_and_ff(nn.Module):
    def __init__(self, chan, require_attn=False):
        super().__init__()
        self.require_attn = require_attn
        self.ila_layer = Residual(Rezero(ImageLinearAttention(chan, norm_queries = True, require_attn_map=require_attn), is_attn=require_attn), is_attn=require_attn)
        self.ff = Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
    def forward(self, x):
        if not self.require_attn:
            return self.ff(self.ila_layer(x))
        else:
            # print("attn_ff_in:", x.shape)
            x, attn = self.ila_layer(x)
            return(self.ff(x), attn)

class Biattn_and_ff(nn.Module):
    def __init__(self, chan, y_dim, z_dim= 64, dropout=[.2,.5], require_attn_map=False):
        super().__init__()
        self.chan = chan
        self.require_attn_map = require_attn_map
        self.ban_layer = BiAttention(chan, y_dim, z_dim, glimpse=1, dropout=[.2,.5])
        self.ff = Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))

    def forward(self, x, q): #query is the style vector.
        # print("input x shape:", x.shape, self.chan) # bchw
        batch_size = x.shape[0]
        assert x.shape[1] == self.chan
        p, _ = self.ban_layer(x.permute(0,2,3,1).view(batch_size, -1, self.chan), q.unsqueeze(1), v_mask=False) # b x 1 x h x w
        p = p.view(batch_size, 1, x.shape[2], x.shape[3])
        x = p * x
        if not self.require_attn_map:
            return self.ff(x)
        else:
            # print("attn_ff_in:", x.shape)
            return(self.ff(x), p)


# attn_and_ff = lambda chan, require_attn: nn.Sequential(*[
    # Residual(Rezero(ImageLinearAttention(chan, norm_queries = True, require_attn_map=require_attn), require_attn)),
    # Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
# ])

# helpers

def exists(val):
    return val is not None

def identity(x): #helper for conposing with transformation pipelines.
    return x

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    if type(images) == tuple:
        batch_size = images[0].shape[0]
        device = images[0].device
    else:
        batch_size = images.shape[0]
        device = images.device
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=device),
                           create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    # if exists(attns):
    #     return [(style_vectorizer(z, attns), num_layers) for z, num_layers in latent_descr]
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def gen_empty_attn(n, image_size, device):

    empty_attn = torch.ones(n, 1, image_size, image_size, device=device)
    empty_attn /= ((image_size) * (image_size))
    return empty_attn

def gen_sampled_attn(n, loader, device):
    attn_maps = next(loader)[1]
    while attn_maps.shape[0] < n:
        attn_maps = torch.cat([attn_maps, next(loader)[1]], dim=0)
    attn_maps = attn_maps[:n]
    return attn_maps.to(device)


# dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, attn_folder = None, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.use_attn = True if attn_folder else False # The directory to store attention maps(or the like).
        self.attn_folder = attn_folder
        self.attn_paths = None

        if self.use_attn:
            self.attn_paths = []
            prefix = Path(f'{folder}').parts
            prefix_attn = Path(f'{self.attn_folder}').parts
            for path in self.paths:
                post = path.parts[len(prefix):]
                attn_path = prefix_attn + post
                self.attn_paths.append(Path(*attn_path))


        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            # RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

        if self.use_attn:
            self.attn_transform = transforms.Compose([
            # transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            # RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor()]
            # transforms.Lambda(expand_greyscale(transparent)
            )


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        transformed = self.transform(img)
        if not self.use_attn:
            return transformed
        attn_path = self.attn_paths[index]
        attn = Image.open(attn_path)
        attn = attn.convert(mode="L")
        transformed_attn = self.attn_transform(attn)
        transformed_attn += EPS
        transformed_attn /= torch.sum(transformed_attn)
        return (transformed, transformed_attn)


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    if type(tensor) == tuple:
        return tuple(torch.flip(t, dims=(3,)) for t in tensor) #flip both generator and attn.
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):

        # images: in normal setting it's a tensor. In attn setting it's a tuple (x, attn_x).

        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)
        # print("images are:", type(images))
        if detach:
            if type(images) == tuple:
                # print("img shape:", images[0].shape, images[1].shape)
                images = tuple(map(lambda x: x.detach(), images))
            else:
                images = images.detach()
        return self.D(images)

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class AttnToStyle(nn.Module):
    # 
    def __init__(self, image_size, emb, lr_mul = 0.1):
        super().__init__()
        layers = []
        # squeeze the img size until it is 4x4.
        filters = 1
        while image_size /2 >= 4:
            layers.extend([nn.Conv2d(filters, 2*filters, 3, padding=1, stride=2), leaky_relu()])
            filters *= 2
            image_size /= 2
        self.net = nn.Sequential(*layers)
        self.lastFC = nn.Sequential(EqualLinear(4*filters, emb, lr_mul), leaky_relu())
    
    def forward(self, x):
        x = self.net(x)
        # print("shape of attn out:", x.shape)
        x = x.reshape(x.shape[0], -1)
        return self.lastFC(x)


class StyleVectorizer(nn.Module):
    def __init__(self, image_size, emb, depth, lr_mul = 0.1):
        super().__init__()
        # self.use_attn = use_attn
        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])
        # self.attn_preprocess = AttnToStyle(image_size, emb, lr_mul)
        # self.final_merge = nn.Sequential(EqualLinear(2*emb, emb, lr_mul), leaky_relu())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        
        x = F.normalize(x, dim=1)
        return self.net(x)
        # assert exists(attn_map)
        # style = F.normalize(x, dim=1)
        # style = self.net(style)
        # attn_emb = self.attn_preprocess(attn_map)
        # x = torch.cat([style, attn_emb], 1)
        # return self.final_merge(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False, merge_with_attn=False):
        super().__init__()
        self.merge_with_attn = merge_with_attn
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        if merge_with_attn:
            input_c = input_channels+1
        else:
            input_c = input_channels

        self.to_style1 = nn.Linear(latent_dim, input_c)
        self.to_noise1 = nn.Linear(1, filters)
        


        self.conv1 = Conv2DMod(input_c, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise, iattn=None):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        if self.merge_with_attn:
            iattn = F.interpolate(iattn, [x.shape[2], x.shape[3]])*(x.shape[2]*x.shape[3])
            iattn = iattn.clamp_(0., 1.)
        # iattn = iattn.permute(0, 2, 3, 1)
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        if self.merge_with_attn:
            x = torch.cat((x, iattn), dim = 1)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        # print("iattn &x", iattn.shape, x.shape)
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, use_attn = False, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        self.use_attn = use_attn

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = Biattn_and_ff(in_chan, self.latent_dim, require_attn_map=use_attn) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent,
                merge_with_attn=self.use_attn
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, input_attn=None, requires_attn_out = False):
        batch_size = styles.shape[0]
        image_size = self.image_size
        attn_outs = []
        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x, style)
                if self.use_attn:
                    attn_outs.append(x[1])
                    x = x[0]
            x, rgb = block(x, rgb, style, input_noise, input_attn)
        if requires_attn_out:
            return rgb, attn_outs
        else:
            return rgb


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [1], use_attn = False, transparent = False, fmap_max = 512):
        super().__init__()
        self.use_attn = use_attn
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = Attn_and_ff(out_chan, use_attn) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, input):
        if self.use_attn:
            x = input[0]
            attn = input[1]
            attn_loss = torch.zeros(x.shape[0]).to(x)
        else:
            x = input
            attn = None
            attn_loss = None

        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)
# 
            if exists(attn_block):
                x = attn_block(x)
                if self.use_attn:
                    # print("attn_xshape", x[0].shape, x[1].shape)
                    curr_attn = F.interpolate(attn, [x[0].shape[2], x[0].shape[3]])
                    # curr_attn = EPS   # to prevent all-zero cases.
                    # curr_attn /= torch.sum(curr_attn) # sum up to 1
                    # print("compute attn device:", x[1].device, curr_attn.device)
                    attn_loss += F.mse_loss(x[1], curr_attn.squeeze(), reduction="none").sum(dim=(1,2)).mean()
                    x = x[0]

            if exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss, attn_loss

class AGGAN(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, use_attn = False, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.use_attn = use_attn

        self.S = StyleVectorizer(image_size, latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, use_attn = use_attn, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, use_attn = use_attn, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(image_size, latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, use_attn = use_attn, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        # We do not use contrastive learner.
        # if cl_reg:
        #     from contrastive_learner import ContrastiveLearner
        #     # experimental contrastive loss discriminator regularization
        #     assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
        #     self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        use_attn = False,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'], 
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.use_attn = use_attn
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        if self.use_attn:
            assert len(self.attn_layers)>0, 'attention-based model must have attention layers.'
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

    @property
    def image_extension(self):
        return 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, use_attn = self.use_attn, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder, attn_folder=None):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob, attn_folder=attn_folder)
        num_workers = num_workers = default(self.num_workers, num_cores)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)
        self.ral = torch.tensor(0.).cuda(self.rank)
        self.fal = torch.tensor(0.).cuda(self.rank)
        self.attn_G = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        # if exists(self.GAN.D_cl):
        #     self.GAN.D_opt.zero_grad()

        #     if apply_cl_reg_to_generated:
        #         for i in range(self.gradient_accumulate_every):
        #             get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        #             style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
        #             noise = image_noise(batch_size, image_size, device=self.rank)

        #             w_space = latent_to_w(self.GAN.S, style)
        #             w_styles = styles_def_to_tensor(w_space)

        #             generated_images = self.GAN.G(w_styles, noise)
        #             self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

        #     for i in range(self.gradient_accumulate_every):
        #         image_batch = next(self.loader).cuda(self.rank)
        #         self.GAN.D_cl(image_batch, accumulate=True)

        #     loss = self.GAN.D_cl.calculate_loss()
        #     self.last_cr_loss = loss.clone().detach().item()
        #     backwards(loss, self.GAN.D_opt, loss_id = 0)

        #     self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        attn_gts = [] # a list to store sampled attn maps.


        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):

            image_batch = next(self.loader)
            # print(batch_size)
            # print("shape from loader:", image_batch[0].shape)
            if self.use_attn:
                attn_gts.append(image_batch[1])
                image_batch = tuple(map(lambda t: t.cuda(self.rank).requires_grad_(), image_batch))
            else:
                image_batch = image_batch.cuda(self.rank)
                image_batch.requires_grad_()

            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            if self.use_attn:
                generated_images = G(w_styles, noise, image_batch[1])
                fake_output, fake_q_loss, fake_attn_loss = D_aug((generated_images.clone().detach(), image_batch[1]), detach = True, **aug_kwargs)
                real_output, real_q_loss, real_attn_loss = D_aug(image_batch, **aug_kwargs)

            else:
                generated_images = G(w_styles, noise)
                fake_output, fake_q_loss, fake_attn_loss = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)
                real_output, real_q_loss, real_attn_loss = D_aug(image_batch, **aug_kwargs)


            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = (F.relu(1 + real_output_loss) + F.relu(1 - fake_output_loss)).mean()
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp
            
            if self.use_attn:
                ral = real_attn_loss.mean()
                fal = fake_attn_loss.mean()
                self.ral += ral.clone().detach().item()
                self.fal += fal.clone().detach().item()

                disc_loss += ral + fal

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')


        self.track(self.ral, "Real_Attn")
        self.track(self.fal, "Fake_Attn")

        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        i = 0
        for _ in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)
            attn_gt = attn_gts[i//len(attn_gts)].cuda(self.rank).requires_grad_() if self.use_attn else None
            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images_ = G(w_styles, noise, input_attn=attn_gt, requires_attn_out=self.use_attn)
            
            if self.use_attn:
                generated_images, attn_out_G = generated_images_
            else:
                generated_images = generated_images_
                attn_out_G = None

            D_in = (generated_images, attn_gt) if self.use_attn else generated_images

            fake_output, _, attn_loss = D_aug(D_in, **aug_kwargs)
            fake_output_loss = fake_output
            # 

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            if self.use_attn:
                attn_loss_G = torch.tensor(0.).cuda(self.rank)
                for attn_out in attn_out_G:
                    attn_gt_interp = F.interpolate(attn_gt, [attn_out.shape[2], attn_out.shape[3]])
                    # print("ATTNG_SHAPE:", attn_gt_interp.shape, attn_out.shape)
                    attn_loss_G += F.mse_loss(attn_out.squeeze(), attn_gt_interp.squeeze(), reduction="none").sum(dim=(1,2)).mean()

                self.attn_G += attn_loss_G.clone().detach().item()

                loss = fake_output_loss.mean() + attn_loss.mean() + attn_loss_G
            else:
                loss = fake_output_loss.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every
            i += 1

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')
        if self.use_attn:
            self.track(self.attn_G, 'Attn_G')
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}, {self.gen_log()}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)
        empty_attn = None
        sampled_attn = None
        if self.use_attn:
            empty_attn = gen_empty_attn(num_rows ** 2, image_size, device=self.rank)
            if exists(self.loader):
                sampled_attn = gen_sampled_attn(num_rows ** 2, self.loader, device=self.rank)
        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, empty_attn, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        if exists(sampled_attn):
            w = map(lambda t: (self.GAN.S(t[0]), t[1]), latents)
            w_truncated = self.truncate_style_defs(w, trunc_psi = self.trunc_psi)
            w_styles = styles_def_to_tensor(w_truncated)
            w_styles_split = w_styles.split(self.batch_size, dim=0)
            noi_split = n.split(self.batch_size, dim=0)
            attn_gt_split = sampled_attn.split(self.batch_size, dim=0)

            gen_images = []
            gen_attns = []

            for w, noi, attn in zip(w_styles_split, noi_split, attn_gt_split):
                gen_img_batch, gen_attn_batch = self.GAN.G(w, noi, attn, True)
                gen_images.append(gen_img_batch)
                gen_attns.append(gen_attn_batch[0])
            generated_images = torch.cat(gen_images, dim=0).clamp_(0., 1.)
            generated_attns = torch.cat(gen_attns, dim=0).clamp_(0., 1.)
            # generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, sampled_attn, trunc_psi = self.trunc_psi)
            torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-sampled.{ext}'), nrow=num_rows)
            torchvision.utils.save_image([attn*255 for attn in sampled_attn], str(self.results_dir / self.name / f'{str(num)}-sampledattn.{ext}'), nrow=num_rows)
            torchvision.utils.save_image([attn*255 for attn in generated_attns], str(self.results_dir / self.name / f'{str(num)}-generatedattn.{ext}'), nrow=num_rows)

        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, empty_attn, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        if exists(sampled_attn):
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, sampled_attn, trunc_psi = self.trunc_psi)
            torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema-sampled.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, empty_attn, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

        if exists(sampled_attn):
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, sampled_attn, trunc_psi = self.trunc_psi)
            torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr-sampled.{ext}'), nrow=num_rows)


    @torch.no_grad()
    def calculate_fid(self, num_batches):
        torch.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)
        attns = []
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch = next(self.loader)
            if self.use_attn:
                img = real_batch[0]
                attns.append(real_batch[1].to(self.rank))
            else:
                img = real_batch
                attns.append(None)
            for k in range(img.size(0)):
                torchvision.utils.save_image(img[k, :, :, :], real_path + '{}.png'.format(k + batch_num * self.batch_size))
            
        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            n = image_noise(self.batch_size, image_size, device=self.rank) 

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, attns[batch_num], trunc_psi = self.trunc_psi)

            for j in range(generated_images.size(0)):
                torchvision.utils.save_image(generated_images[j, :, :, :], str(Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, self.rank, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, attn_gt=None, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        if exists(attn_gt):
            generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi, attn_gt)
        else:
            generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, attn_map=None, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles 

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, attn_map, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def gen_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]

        data_highacc = [
            ('R_Attn', self.ral),
            ('F_Attn', self.fal),
            ('G_Attn', self.attn_G)]
        data_highacc = [d for d in data_highacc if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        log_highacc = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.7f}', data_highacc))
        return " ".join([log, ' | ', log_highacc])

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])

class ModelLoader:
    def __init__(self, *, base_dir, name = 'default', load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, attn_map, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.S(noise, attn_map)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.G.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.G(w_tensors, noise)
        images.clamp_(0., 1.)
        return images
