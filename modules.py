from .util import mask_, d, slice_diag

import torch
from torch import nn
import torch.functional as F

import random, math, sys

class SelfAttention(nn.Module):
    """
    Implementation of multi-head self attention
    """

    def __init__(self, emb, heads=8,mask=False, kqnorm=False, scalefactor=None):

        """
        :param emb: dimension of each vector of the sequence
        :param heads: number of heads being used in multi-headed attention
        :param mask: if autoregressive mask is applied or not
        :param kqnorm: if normalization is applied to keys and queries
        :param scalefactor: multiplier for attention weights(if none then '1/sqrt(emb/heads)' is used)
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by number of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb//heads
        #breaking 'heads' into different chunks and then using them for different self attention heads

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

        self.scalefactor = 1/math.sqrt(emb//heads) if scalefactor is None else scalefactor