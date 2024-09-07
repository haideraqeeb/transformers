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

    def forward(self, x):
        b, t, e = x.size()

        """
        :b: batch size
        :t: embedding tokens in the sequence(vectors)
        :e: dimension of each vector
        """
        h = self.heads

        assert e == self.emb, f'Input embedding dimension ({e}) should match layer embedding dimension ({self.emb})'

        s = e//h

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        #dividing into a number of chunks to work with multiple self attention heads

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        #folding the heads into the batch dimension

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        #computing the scaled dot product self attention

        dot = torch.bmm(queries, keys.transpose(1, 2))        
        dot = dot * self.scalefactor

        assert dot.size() == (b * h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)

        y = torch.bmm(dot, values).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().views(b, t, s * h)

        return self.unifyheads(out)
    
class TransformerBlock(nn.Module):
    """
    Straightforward transfomer module
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0,
                sa_kwargs={}):
        
        super.__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask, **sa_kwargs)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.drop(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x