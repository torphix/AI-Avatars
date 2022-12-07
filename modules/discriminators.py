
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class DiscriminatorP(torch.nn.Module):
    def __init__(self, in_d, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(in_d, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPatchDiscriminator(torch.nn.Module):
    def __init__(self, in_d):
        super(MultiPatchDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(in_d),
            DiscriminatorP(in_d),
            DiscriminatorP(in_d),
            DiscriminatorP(in_d),
            DiscriminatorP(in_d),
        ])

    def forward(self, x):
        outs = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            out, fmap = d(x)
            outs.append(out)
            fmaps.append(fmap)

        return out, fmaps


class DiscriminatorS(torch.nn.Module):
    def __init__(self, in_d, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(in_d, 128, 10, 1, padding=7)),
            norm_f(nn.Conv2d(128, 256, 15, 2, groups=4, padding=7)),
            norm_f(nn.Conv2d(256, 256, 15, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv2d(256, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, in_d):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(in_d, use_spectral_norm=True),
            DiscriminatorS(in_d),
            DiscriminatorS(in_d),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool2d(4, 2, padding=2),
            nn.AvgPool2d(4, 2, padding=2)
        ])

    def forward(self, x):
        outs = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
            x_d, fmap_out = d(x)
            outs.append(x_d), fmaps.append(fmap_out)

        return outs, fmaps