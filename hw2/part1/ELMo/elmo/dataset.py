import pickle
from operator import itemgetter
import ipdb
from box import Box
import torch
from torch.utils.data import Dataset


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


class ElmoDataset(Dataset):
    def __init__(self, corpus_path, id2word_path):
        self.data = self.load_pkl(corpus_path)
        self.id2word = self.load_pkl(id2word_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]

        # char process pipeline
        # char_line = []
        # for token_idx in line:
        #     if token_idx == BOS_INDEX:
        #         char_line.append([CHAR_BOW_INDEX])
        #     elif token_idx == EOS_INDEX:
        #         char_line.append([CHAR_EOW_INDEX])
        #     else:
        #         word = self.id2word[token_idx]
        #         char_line.append([ord(char) if 0 <= ord(char) <= 255 else CHAR_OOV_INDEX
        #                           for char in list(word)])
        char_line = ElmoDataset.sent_idx_to_char_map(line, self.id2word)

        forward_char_line = char_line
        backward_char_line = list(reversed(forward_char_line))

        # word process pipeline
        forward_line = line
        backward_line = list(reversed(forward_line))
        length = len(forward_line)

        return Box(
            {'forward_line': forward_line,
             'backward_line': backward_line,
             'length': length,
             'forward_char_line': forward_char_line,
             'backward_char_line': backward_char_line}
        )

    @staticmethod
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def add_bos_eos(line):
        return [BOS_INDEX] + line + [EOS_INDEX]

    @staticmethod
    def sent_idx_to_char_map(idx_sent, id2word):
        char_map = []
        for token_idx in idx_sent:
            if token_idx == BOS_INDEX:
                char_map.append([CHAR_BOW_INDEX])
            elif token_idx == EOS_INDEX:
                char_map.append([CHAR_EOW_INDEX])
            else:
                word = id2word[token_idx]
                char_map.append([ord(char) if 0 <= ord(char) <= 255 else CHAR_OOV_INDEX
                                 for char in list(word)])
        return char_map

    @staticmethod
    def pad_word_map(word_maps):
        max_length = max(list(map(len, word_maps)))
        # word_maps = [words[-max_length:] for words in word_maps]
        word_maps = [words + [PAD_INDEX] * (max_length - len(words)) for words in word_maps]
        word_maps = torch.tensor(word_maps, dtype=torch.int64)
        return word_maps

    @staticmethod
    def pad_char_map(char_maps, limit_word_len=15):
        sent_len = max(list(map(len, char_maps)))
        word_len = min(max(list(map(lambda x: max([len(l) for l in x]), char_maps))), limit_word_len)
        char_maps = [[word[-word_len:]
                      for word in c_map]
                     for c_map in char_maps]
        char_maps = [[word + [CHAR_PAD_INDEX] * (word_len - len(word))
                      for word in c_map]
                     for c_map in char_maps]
        char_maps = [c_map + [[CHAR_PAD_INDEX] * word_len] * (sent_len - len(c_map))
                     for c_map in char_maps]
        char_maps = torch.tensor(char_maps, dtype=torch.int64)
        return char_maps

    @staticmethod
    def collect_fn(batch, length_descending=True):
        if length_descending:
            batch.sort(key=itemgetter('length'), reverse=True)
        return Box(
            {'forward': {'words': ElmoDataset.pad_word_map([x.forward_line for x in batch]),
                         'chars': ElmoDataset.pad_char_map([x.forward_char_line for x in batch])},
             'backward': {'words': ElmoDataset.pad_word_map([x.backward_line for x in batch]),
                          'chars': ElmoDataset.pad_char_map([x.backward_char_line for x in batch])},
             'word_length': torch.tensor([x.length for x in batch], dtype=torch.int64)}
        )

    @staticmethod
    def move_batch_to_device(batch, device):
        batch = Box(batch)
        batch.forward.words = batch.forward.words.to(device)
        batch.forward.chars = batch.forward.chars.to(device)
        batch.backward.words = batch.backward.words.to(device)
        batch.backward.chars = batch.backward.chars.to(device)
        batch.word_length = batch.word_length.to(device)
        return batch
