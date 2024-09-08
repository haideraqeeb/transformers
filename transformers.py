import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

class GTransformer(nn.Module):
    """
    Transformer for generating sequence
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens):

        """
        :param emb: size of embedding vectors in the sequence
        :param heads: number of attention heads in each transformer block
        :param depth: number of transformer blocks in the transformer
        :param seq_length: length of the input sequence sent to the transformer
        :param num_tokens: number of tokens in the sequence
        """

        super.__init__()

        self.num_tokens = num_tokens
        
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=True, seq_length=seq_length, pos_embedding=self.pos_embedding))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: the input sequence of tokens
        :return: predicted log probability vectors for each token based of the preceding tokens
        """

        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arrange(t, device=d()))[None, :, :].expand(b, t, e)

        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)