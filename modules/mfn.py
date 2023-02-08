#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize

    
# GaborLayer and GaborNet from https://github.com/addy1997/mfn-pytorch/blob/main/model/MultiplicativeFilterNetworks.py
class GaborLayer(nn.Module):
    def __init__(self, in_dim, out_dim, padding, alpha, beta=1.0, bias=False):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # Bias parameters start in zeros
        #self.bias = nn.Parameter(torch.zeros(self.responses)) if bias else None

    def forward(self, input):
        norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))


class INR(nn.Module):
    def __init__(self, in_features=2, hidden_features=256,
                 hidden_layers=4, out_features=1, 
                 outermost_linear=True, first_omega_0=0,
                 hidden_omega_0=0, scale=1, pos_encode=False,
                 sidelength=1, fn_samples=None, use_nyquist=None):
        super(INR, self).__init__()

        self.k = hidden_layers+1
        self.gabon_filters = nn.ModuleList([GaborLayer(in_features, hidden_features, 0, alpha=6.0 / self.k) for _ in range(self.k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_features, hidden_features) for _ in range(self.k - 1)] + [torch.nn.Linear(hidden_features, out_features)])

        for lin in self.linear[:self.k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, x):

        # Recursion - Equation 3
        zi = self.gabon_filters[0](x[0, ...])  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x[0, ...])
            # Eq 3.b

        return self.linear[self.k - 1](zi)[None, ...]  # Eq 3.c