import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image

from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        if 'actv' in config:
            self.actv = config['actv']
        else:
            self.actv = 'clamp'
        if 'ref' in config:
            self.ref = config['ref']
        else:
            self.ref = False
        if 'addmaskcoarse' in config:
            self.addmaskcoarse = config['addmaskcoarse']
        else:
            self.addmaskcoarse = False
        if 'addmask' in config:
            self.addmask = config['addmask']
        else:
            self.addmask = False
        if 'n_blocks' in config:
            self.n_blocks = config['n_blocks']
        else:
            self.n_blocks = 2
        if 'gated' in config:
            self.gated = config['gated']
        else:
            self.gated = False
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        if not self.addmaskcoarse:
            self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.n_blocks, self.gated,
                                                    self.use_cuda, self.device_ids)
        else:
            self.coarse_generator = CoarseGenerator_WMask(self.input_dim, self.cnum, self.n_blocks, self.gated,
                                                          self.use_cuda, self.device_ids)
        if self.ref:
            self.fine_generator = FineGenerator_WRef(self.input_dim, self.cnum, self.n_blocks, self.gated,
                                                     self.use_cuda, self.device_ids)
            if self.n_blocks > 2:
                print('Not implemented!!!!!')
                xxxxxxxxxx
        elif self.addmask:
            self.fine_generator = FineGenerator_WMask(self.input_dim, self.cnum, self.n_blocks, self.gated,
                                                      self.use_cuda, self.device_ids)
            if self.n_blocks > 2:
                print('Not implemented!!!!!')
                xxxxxxxxxx
        else:
            self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.n_blocks, self.gated,
                                                self.use_cuda, self.device_ids)

    def forward(self, x, mask, ref=None):
        if self.addmaskcoarse:
            x_stage1 = self.coarse_generator(x, mask, ref)
        else:
            x_stage1 = self.coarse_generator(x, mask)
        x_stage1 = self.forward_actv(x_stage1)
        if self.ref or self.addmask:
            x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask, ref)
        else:
            x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        x_stage2 = self.forward_actv(x_stage2)
        return x_stage1, x_stage2, offset_flow

    def forward_actv(self, x):
        if self.actv == 'tanh':
            return torch.tanh(x)
        else:
            return torch.clamp(x, -1., 1.)


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, n_blocks=2, gated=False, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.n_blocks = n_blocks
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.gated = gated

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, gated=gated)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1, gated=gated)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1, gated=gated)
        for i in range(2, n_blocks):
            setattr(self, f'conv3_{i}', gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated))
            setattr(self, f'conv4_downsample_{i}', gen_conv(cnum*4, cnum*4, 3, 2, 1, gated=gated))

        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2, gated=gated)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4, gated=gated)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8, gated=gated)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16, gated=gated)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1, gated=gated)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated)
        for i in range(2, n_blocks):
            setattr(self, f'conv13_{i}', gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated))
            setattr(self, f'conv14_{i}', gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated))

        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1, gated=gated)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1, gated=gated)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none', gated=gated)

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        for i in range(2, self.n_blocks):
            x = getattr(self, f'conv3_{i}')(x)
            x = getattr(self, f'conv4_downsample_{i}')(x)

        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        for i in range(2, self.n_blocks):
            x = getattr(self, f'conv13_{i}')(x)
            x = getattr(self, f'conv14_{i}')(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256

        return x


class CoarseGenerator_WMask(CoarseGenerator):
    def __init__(self, input_dim, cnum, n_blocks=2, gated=False, use_cuda=True, device_ids=None):
        super(CoarseGenerator_WMask, self).__init__(input_dim, cnum, n_blocks=n_blocks, gated=gated, use_cuda=use_cuda,
                                                  device_ids=device_ids)
        # 4 x 256 x 256
        self.conv1 = gen_conv(input_dim + 3, cnum, 5, 1, 2, gated=gated)
        # attention branch
        # 4 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 3, cnum, 5, 1, 2, gated=gated)

    def forward(self, x, mask, ref):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        ref = ref * mask
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask, ref], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        for i in range(2, self.n_blocks):
            x = getattr(self, f'conv3_{i}')(x)
            x = getattr(self, f'conv4_downsample_{i}')(x)

        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        for i in range(2, self.n_blocks):
            x = getattr(self, f'conv13_{i}')(x)
            x = getattr(self, f'conv14_{i}')(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256

        return x


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, n_blocks=2, gated=False, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.n_blocks = n_blocks
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.gated = gated

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, gated=gated)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1, gated=gated)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1, gated=gated)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1, gated=gated)
        for i in range(2, n_blocks):
            setattr(self, f'conv3_{i}', gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated))
            setattr(self, f'conv4_downsample_{i}', gen_conv(cnum*2, cnum*2, 3, 2, 1, gated=gated))

        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1, gated=gated)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2, gated=gated)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4, gated=gated)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8, gated=gated)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16, gated=gated)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2, gated=gated)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1, gated=gated)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1, gated=gated)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1, gated=gated)
        for i in range(2, n_blocks):
            setattr(self, f'pmconv3_{i}', gen_conv(cnum * 4, cnum * 4, 3, 1, 1, gated=gated))
            setattr(self, f'pmconv4_downsample_{i}', gen_conv(cnum * 4, cnum * 4, 3, 2, 1, gated=gated))

        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu', gated=gated)
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       downsample=2**self.n_blocks,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1, gated=gated)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1, gated=gated)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1, gated=gated)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated)
        for i in range(2, n_blocks):
            setattr(self, f'allconv13_{i}', gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated))
            setattr(self, f'allconv14_{i}', gen_conv(cnum*2, cnum*2, 3, 1, 1, gated=gated))
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1, gated=gated)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1, gated=gated)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none', gated=gated)

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        for i in range(2, self.n_blocks):
            x = getattr(self, f'conv3_{i}')(x)
            x = getattr(self, f'conv4_downsample_{i}')(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        for i in range(2, self.n_blocks):
            x = getattr(self, f'pmconv3_{i}')(x)
            x = getattr(self, f'pmconv4_downsample_{i}')(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        for i in range(2, self.n_blocks):
            x = getattr(self, f'allconv13_{i}')(x)
            x = getattr(self, f'allconv14_{i}')(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)

        return x, offset_flow


class FineGenerator_WRef(FineGenerator):
    def forward(self, xin, x_stage1, mask, ref):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        zeros = torch.zeros(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
            zeros = zeros.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x

        # attention branch
        ref = torch.cat([ref, ones, zeros], dim=1)
        xnowref = torch.cat([xnow, ref], dim=2)
        xnowref = self.pmconv1(xnowref)
        xnowref = self.pmconv2_downsample(xnowref)
        xnowref = self.pmconv3(xnowref)
        xnowref = self.pmconv4_downsample(xnowref)
        xnowref = self.pmconv5(xnowref)
        ref = self.pmconv6(xnowref)
        x = ref[:, :, :int(xin.size(2)/4), :]
        mask = torch.cat((mask, zeros), dim=2)

        x, offset_flow = self.contextul_attention(x, ref, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)

        return x, offset_flow


class FineGenerator_WMask(FineGenerator):
    def __init__(self, input_dim, cnum, n_blocks=2, gated=False, use_cuda=True, device_ids=None):
        super(FineGenerator_WMask, self).__init__(input_dim, cnum, n_blocks=n_blocks, gated=gated, use_cuda=use_cuda, device_ids=device_ids)
        # 4 x 256 x 256
        self.conv1 = gen_conv(input_dim + 3, cnum, 5, 1, 2, gated=gated)
        # attention branch
        # 4 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 3, cnum, 5, 1, 2, gated=gated)

    def forward(self, xin, x_stage1, mask, ref):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        ref = ref * mask
        xnow = torch.cat([x1_inpaint, ones, mask, ref], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)

        return x, offset_flow


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10, downsample=4,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.downsample = downsample
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(self.downsample*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')

        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            """if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)"""
            offset = torch.cat([offset//int_bs[3], offset%int_bs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*self.downsample, mode='nearest')

        return y, flow


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class GlobalPatchDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalPatchDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = PatchDisConvModule(self.input_dim, self.cnum)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)

        return x


class PatchDisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(PatchDisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim+1, cnum, 5, 2, 2, weight_norm='sn')
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2, weight_norm='sn')
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2, weight_norm='sn')
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2, weight_norm='sn')
        self.conv5 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2, weight_norm='sn')
        self.conv6 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2, weight_norm='sn')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu', gated=False):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, gated=gated)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu', weight_norm='none'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, weight_norm=weight_norm)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False, gated=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.gated = gated
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
            if self.gated:
                print('Transposed Gated not implemented!')
                xxxxxxx
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)
            if self.gated:
                self.mask_conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                      padding=conv_padding, dilation=dilation,
                                      bias=self.use_bias)
                self.sigmoid = torch.nn.Sigmoid()


        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x0):
        if self.pad:
            x0 = self.pad(x0)
            x = self.conv(x0)
        else:
            x = self.conv(x0)
        if self.gated:
            mask = self.mask_conv(x0)
            mask = self.sigmoid(mask)
            x = x * mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
