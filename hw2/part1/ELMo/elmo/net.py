import ipdb
from box import Box
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .embedding import TokenEmbedding


PAD_INDEX = 1


class LayerNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, tensor):
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta


class ELMo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embedding.char.projection_size + config.embedding.word.embedding_dim
        self.embedding = TokenEmbedding(embedding_npy=config.embedding.word.embedding_npy,
                                        **config.embedding.char)
        self.vocab_size = self.embedding.word_embedding.num_embeddings

        self.dropout = nn.Dropout(p=config.model.dropout)
        self.rnn = nn.ModuleDict({
            'first_forward': nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=config.model.rnn.hidden_size[0],
                batch_first=True),
            'first_backward': nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=config.model.rnn.hidden_size[0],
                batch_first=True),
            'second_forward': nn.LSTM(
                input_size=config.model.rnn.proj_size[0],
                hidden_size=config.model.rnn.hidden_size[1],
                batch_first=True),
            'second_backward': nn.LSTM(
                input_size=config.model.rnn.proj_size[0],
                hidden_size=config.model.rnn.hidden_size[1],
                batch_first=True)
        })
        self.proj = nn.ModuleDict({
            'first_forward': nn.Linear(config.model.rnn.hidden_size[0], config.model.rnn.proj_size[0]),
            'first_backward': nn.Linear(config.model.rnn.hidden_size[0], config.model.rnn.proj_size[0]),
            'second_forward': nn.Linear(config.model.rnn.hidden_size[1], config.model.rnn.proj_size[1]),
            'second_backward': nn.Linear(config.model.rnn.hidden_size[1], config.model.rnn.proj_size[1])
        })
        self.adaptive_sm = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=config.model.rnn.proj_size[1],
            n_classes=self.vocab_size,
            cutoffs=((self.vocab_size // config.model.adaptive_softmax.factor) *
                     np.array(config.model.adaptive_softmax.cutoff_coefs, dtype=np.int)).tolist(),
            div_value=config.model.adaptive_softmax.div_value,
            head_bias=config.model.adaptive_softmax.head_bias
        )

        self.layer_norm_f_f = LayerNorm((config.model.rnn.proj_size[0]))
        self.layer_norm_f_s = LayerNorm((config.model.rnn.proj_size[0]))
        self.layer_norm_b_f = LayerNorm((config.model.rnn.proj_size[0]))
        self.layer_norm_b_s = LayerNorm((config.model.rnn.proj_size[0]))

    def forward(self, batch, get_vector=False):
        if get_vector:
            vector = []
        # LSTM + Proj_linear
        # forward
        f_embed = self.embedding(batch.forward)  # B x Len x embed_dim
        if get_vector:
            vector.append(f_embed[:, 1:, self.config.embedding.word.embedding_dim:].repeat(1, 1, 2))

        f_seq = pack_padded_sequence(f_embed, batch.word_length, batch_first=True)
        f_output, (_, _) = self.rnn['first_forward'](f_seq)
        f_output, _ = pad_packed_sequence(f_output, batch_first=True)
        f_output = self.proj['first_forward'](f_output)
        f_output = F.leaky_relu(f_output)
        if self.config.model.skip_connection:
            f_output = f_output + f_embed[:, :, self.config.embedding.word.embedding_dim:]
        f_output = self.layer_norm_f_f(f_output)
        f_output = self.dropout(f_output)
        if get_vector:
            vector.append(f_output[:, 1:, :])
        f_output = pack_padded_sequence(f_output, batch.word_length, batch_first=True)
        f_output, (_, _) = self.rnn['second_forward'](f_output)
        f_output, _ = pad_packed_sequence(f_output, batch_first=True)
        f_output = self.proj['second_forward'](f_output)
        f_output = F.leaky_relu(f_output)
        f_output = self.layer_norm_f_s(f_output)
        f_output = self.dropout(f_output)
        if get_vector:
            vector.append(f_output[:, 1:, :])

        # backward
        b_embed = self.embedding(batch.backward)  # B x Len x embed_dim
        b_seq = pack_padded_sequence(b_embed, batch.word_length, batch_first=True)
        b_output, (_, _) = self.rnn['first_forward'](b_seq)
        b_output, _ = pad_packed_sequence(b_output, batch_first=True)
        b_output = self.proj['first_forward'](b_output)
        b_output = F.leaky_relu(b_output)
        if self.config.model.skip_connection:
            b_output = b_output + b_embed[:, :, self.config.embedding.word.embedding_dim:]
        b_output = self.layer_norm_b_f(b_output)
        b_output = self.dropout(b_output)
        if get_vector:
            out = b_output[:, 1:, :]
            for i in range(len(out)):
                out[i, :(batch.word_length[i]-1), :] = torch.flip(out[i, :(batch.word_length[i]-1), :], [0])
            vector.append(out)
        b_output = pack_padded_sequence(b_output, batch.word_length, batch_first=True)
        b_output, (_, _) = self.rnn['second_forward'](b_output)
        b_output, _ = pad_packed_sequence(b_output, batch_first=True)
        b_output = self.proj['second_backward'](b_output)
        b_output = F.leaky_relu(b_output)
        b_output = self.layer_norm_b_s(b_output)
        b_output = self.dropout(b_output)
        if get_vector:
            out = b_output[:, 1:, :]
            for i in range(len(out)):
                out[i, :(batch.word_length[i]-1), :] = torch.flip(out[i, :(batch.word_length[i]-1), :], [0])
            vector.append(out)

            vector[1] = torch.cat((vector[1], vector[3]), dim=-1)
            vector[2] = torch.cat((vector[2], vector[4]), dim=-1)
            del vector[3:]
            vector = torch.stack(vector)
            return vector

        # adaptive softmax
        f_ans = batch.forward.words[:, 1:].contiguous().view(-1)
        f_out, f_loss = self.adaptive_sm(f_output[:, :-1, :].contiguous().view(-1, f_output.size(-1)),
                                         f_ans)
        f_mask = f_ans.ne(PAD_INDEX).type(torch.float).view(-1)
        f_loss = (-f_out * f_mask).sum() / f_mask.sum()

        b_ans = batch.backward.words[:, 1:].contiguous().view(-1)
        b_out, b_loss = self.adaptive_sm(b_output[:, :-1, :].contiguous().view(-1, b_output.size(-1)),
                                         b_ans)
        b_mask = b_ans.ne(PAD_INDEX).type(torch.float).view(-1)
        b_loss = (-b_out * b_mask).sum() / b_mask.sum()

        # with torch.no_grad():
        #     f_accu = (self.adaptive_sm.predict(f_output.detach().view(-1, f_output.size(-1)))\
        #                  .eq(batch.forward.words.view(-1)).type(torch.float) * f_mask).sum()\
        #              / f_mask.sum()
        #     b_accu = (self.adaptive_sm.predict(b_output.detach().view(-1, b_output.size(-1)))\
        #                  .eq(batch.backward.words.view(-1)).type(torch.float) * b_mask).sum()\
        #              / b_mask.sum()
        #     accu = (f_accu + b_accu) / 2

        return f_loss, b_loss, -1
