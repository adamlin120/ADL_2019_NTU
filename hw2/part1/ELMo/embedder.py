import torch
from torch.nn import functional as F
import yaml
import numpy as np
from box import Box
from .elmo import net
from .elmo.dataset import ElmoDataset


OOV_INDEX = 0
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
OOV_TOKEN = '<OOV>'
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'

CHAR_PAD_INDEX = 256
CHAR_OOV_INDEX = 257
CHAR_BOW_INDEX = 258
CHAR_EOW_INDEX = 259


class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim,
                 elmo_config, elmo_ckpt_path,
                 layer_norm,
                 dropout,
                 device):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim

        self.layer_norm = layer_norm
        self.device = device
        self.dropout = dropout
        with open(elmo_config, 'rb') as f:
            elmo_config = Box(yaml.load(f))
        self.elmo = getattr(net, elmo_config.model.net)(elmo_config)
        self.elmo.load_state_dict(torch.load(elmo_ckpt_path)['model_state_dict'])
        self.elmo.eval()
        self.elmo.to(self.device)
        self.word2idx = ElmoDataset.load_pkl(elmo_config.data.word2idx)
        self.idx2word = ElmoDataset.load_pkl(elmo_config.data.idx2word)

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        ori_lens = list(map(len, sentences))
        # convert to idx
        f_idx_sents = [[self.word2idx.get(token, OOV_INDEX)
                        for i, token in enumerate(token_sent) if i+1 <= max_sent_len]
                       for token_sent in sentences]
        b_idx_sents = [list(reversed(sent)) for sent in f_idx_sents]

        # add forward: box, backward: eos
        f_idx_sents = [[BOS_INDEX] + sent for sent in f_idx_sents]
        b_idx_sents = [[EOS_INDEX] + sent for sent in b_idx_sents]

        # create char map
        f_char_maps = ElmoDataset.pad_char_map(
            [ElmoDataset.sent_idx_to_char_map(idx_sent, self.idx2word) for idx_sent in f_idx_sents])
        b_char_maps = ElmoDataset.pad_char_map(
            [ElmoDataset.sent_idx_to_char_map(idx_sent, self.idx2word) for idx_sent in b_idx_sents])

        lengths = torch.tensor(list(map(len, f_idx_sents)), dtype=torch.int64)

        # make batch
        f_idx_sents = ElmoDataset.pad_word_map(f_idx_sents)
        b_idx_sents = ElmoDataset.pad_word_map(b_idx_sents)
        batch = Box(({
            'forward': {'words': f_idx_sents,
                        'chars': f_char_maps},
            'backward': {'words': b_idx_sents,
                         'chars': b_char_maps},
            'word_length': lengths
        }))

        # sort by length
        sorted_length, sort_arg = torch.sort(lengths, descending=True)
        unsort_arg = torch.argsort(sort_arg)

        batch.forward.words = batch.forward.words[sort_arg].to(self.device)
        batch.forward.chars = batch.forward.chars[sort_arg].to(self.device)
        batch.backward.words = batch.backward.words[sort_arg].to(self.device)
        batch.backward.chars = batch.backward.chars[sort_arg].to(self.device)
        batch.word_length = batch.word_length[sort_arg].to(self.device)

        # pass model
        vectors = self.elmo.forward(batch, get_vector=True).detach().cpu()
        vectors = F.dropout(vectors, self.dropout)
        vectors = vectors.numpy().astype(np.float32)
        vectors = np.swapaxes(vectors, 0, 1)
        vectors = np.swapaxes(vectors, 1, 2)
        # unsort by length
        vectors = vectors[unsort_arg]
        if self.layer_norm:
            mean = np.mean(vectors, (1, 3), keepdims=True)
            std = np.std(vectors, (1, 3), keepdims=True)
            vectors = (vectors - mean) / std
        # remove bos eos
        # vectors = vectors[:, 1:, :, :]
        # reverse backward
        # for i, len in enumerate(ori_lens):
        #     vectors[i, :len, :, :] = vectors[i, :len, :, :][:, :, -1]

        return vectors
