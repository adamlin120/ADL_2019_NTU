import numpy as np
import ipdb
import torch
from torch import nn
from .char_embedding import CharEmbedding


class TokenEmbedding(nn.Module):
    def __init__(self, embedding_npy, **char_embed_args):
        super().__init__()

        self.word_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(np.load(embedding_npy)),
            freeze=True
        )
        self.char_embedding = CharEmbedding(**char_embed_args)

    def forward(self, input):
        with torch.no_grad():
            word_embed = self.word_embedding(input.words).requires_grad_(False)
        char_embed = self.char_embedding(input.chars)
        return torch.cat((word_embed, char_embed), dim=2)
