# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: network architecture


import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class GatedConv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True):
        super(GatedConv2D, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch*2, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_ch*2)

        nn.init.normal_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.conv(x)
        if self.batch_norm:
            h = self.bn(h)
        h = h.split(h.size(1)//2, dim=1)

        return h[0] * torch.sigmoid(h[1])


class GatedDeconv2D(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True):
        super(GatedDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_ch, output_ch*2, kernel_size, stride, padding,
                                         groups=groups, bias=bias, dilation=dilation)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_ch*2)

        nn.init.normal_(self.deconv.weight)
        if bias:
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        h = self.deconv(x)
        if self.batch_norm:
            h = self.bn(h)
        h = h.split(h.size(1)//2, dim=1)

        return h[0] * torch.sigmoid(h[1])


def concat_xy(x, y):
    n_h, n_w = x.shape[2:4]
    return torch.cat((x, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_h, n_w)), dim=1)


class Encoder(nn.Module):
    def __init__(self, n_freq, n_label, type="1D"):
        super(Encoder, self).__init__()
        self.type = type
        self.conv1 = GatedConv2D(n_freq+n_label, n_freq//2, (1, 5), (1, 1), (0, 2))
        self.conv2 = GatedConv2D(n_freq//2+n_label, n_freq//4, (1, 4), (1, 2), (0, 1))
        self.conv3 = nn.Conv2d(n_freq//4+n_label, n_freq//8*2, (1, 4), (1, 2), (0, 1))

    def forward(self, x, l):
        if self.type == "1D":
            x = x.permute(0, 2, 1, 3)
        h = self.conv1(concat_xy(x, l))
        h = self.conv2(concat_xy(h, l))
        h = self.conv3(concat_xy(h, l))
        mu, logvar = h.split(h.size(1)//2, dim=1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, n_freq, n_label, type="1D"):
        super(Decoder, self).__init__()
        self.type = type
        self.deconv1 = GatedDeconv2D(n_freq//8+n_label, n_freq//4, (1, 4), (1, 2), (0, 1))
        self.deconv2 = GatedDeconv2D(n_freq//4+n_label, n_freq//2, (1, 4), (1, 2), (0, 1))
        self.deconv3 = nn.ConvTranspose2d(n_freq//2+n_label, n_freq, (1, 5), (1, 1), (0, 2))

    def forward(self, z, l):
        h = self.deconv1(concat_xy(z, l))
        h = self.deconv2(concat_xy(h, l))
        h = self.deconv3(concat_xy(h, l))

        if self.type == "1D":
            h = h.permute(0, 2, 1, 3)

        return torch.clamp(h, min=-80.)


class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(), device=mu.device))

        return mu + torch.exp(logvar / 2) * eps  # Reparameterization trick

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self, x, l):
        l = l.unsqueeze(0).repeat(x.size(0), 1)
        self.z_mu, self.z_logvar = self.encoder(x, l)
        z = self.sample_z(self.z_mu, self.z_logvar)
        self.x_logvar = self.decoder(z, l)

        return self.x_logvar

    def loss(self, x):
        # closed-form of KL divergence between N(mu, var) and N(0, I)
        kl_loss_el = .5 * (self.z_mu.pow(2) + self.z_logvar.exp() - self.z_logvar - 1)
        kl_loss = torch.sum(kl_loss_el) / x.numel()

        # negative log-likelihood of complex proper Gaussian distribution
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(), device=self.x_logvar.device)

        nll_real = self.gaussian_nll(x, x_zero, logvar, "sum") / x.numel()  # real part
        nll_imag = self.gaussian_nll(x_zero, x_zero, logvar, "sum") / x.numel()  # imaginary part
        nll_loss = nll_real + nll_imag

        loss = kl_loss + nll_loss

        return loss, kl_loss, nll_loss


class SourceModel(nn.Module):
    def __init__(self, decoder, z, l):
        super(SourceModel, self).__init__()
        self.decoder = decoder
        self.z_layer = z
        self.l_layer = l
        self.softmax = nn.Softmax(dim=1)

    def gaussian_nll(self, x, mu, logvar, reduce="sum"):
        x_prec = torch.exp(-logvar)
        x_diff = x - mu
        x_power = -.5 * (x_diff * x_diff) * x_prec
        loss = (logvar + math.log(2 * math.pi)) / 2 - x_power
        if reduce == "sum":
            gaussian_nll = torch.sum(loss)
        elif reduce == "mean":
            gaussian_nll = torch.mean(loss)

        return gaussian_nll

    def forward(self):
        label = self.softmax(self.l_layer)
        self.x_logvar = self.decoder(self.z_layer, label)

        return None

    def loss(self, x):
        self.forward()
        logvar = self.x_logvar - math.log(2.)
        x_zero = torch.zeros(self.x_logvar.size(), device=self.x_logvar.device, dtype=torch.float)
        z_zero = torch.zeros(self.z_layer.size(), device=self.z_layer.device, dtype=torch.float)

        kl_loss = self.gaussian_nll(self.z_layer, z_zero, z_zero, "sum") / x.numel()
        nll_real = self.gaussian_nll(x, x_zero, logvar, "sum") / x.numel()  # real part
        nll_imag = self.gaussian_nll(x_zero, x_zero, logvar, "sum") / x.numel()  # imaginary part
        nll_loss = nll_real + nll_imag

        loss = nll_loss + kl_loss

        return loss

    def get_power_spec(self, cpu=True):
        label = self.softmax(self.l_layer)
        if cpu is True:
            return np.squeeze(np.exp(self.decoder(self.z_layer, label).detach().to("cpu").numpy()), axis=1)
        else:
            return torch.squeeze(torch.exp(self.decoder(self.z_layer, label)), dim=1)

    def get_label(self, cpu=True):
        label = self.softmax(self.l_layer)
        if cpu is True:
            return label.detach().to("cpu").numpy()
        else:
            return label
