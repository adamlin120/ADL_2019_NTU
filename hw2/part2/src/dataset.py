import re
import logging
import pandas
from box import Box
import ipdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_pretrained_bert
#from allennlp.modules.elmo import batch_to_ids


name2TOKENIZER = {'bert-base-uncased': 'BertTokenizer',
                  'bert-large-uncased': 'BertTokenizer',
                  'bert-base-cased': 'BertTokenizer',
                  'bert-large-cased': 'BertTokenizer',
                  'bert-base-multilingual-uncased': 'BertTokenizer',
                  'bert-base-multilingual-cased': 'BertTokenizer',
                  'bert-base-chinese': 'BertTokenizer',
                  'openai-gpt': 'OpenAIGPTTokenizer',
                  'transfo-xl-wt103': 'TransfoXLTokenizer',
                  'gpt2': 'GPT2Tokenizer',
                  'elmo': 'Basic',
                  }


class MyDataset(Dataset):
    def __init__(self, csv_file, supervised, pre_train, remove_punc=False):
        self.pre_train = pre_train
        self.remove_punc = remove_punc
        tokenizer_args = dict(do_lower_case=True) \
            if pre_train.split('-')[-1] == 'uncased' else dict()
        
        self.tokenizer_type = name2TOKENIZER[pre_train]
        if pre_train.startswith('elmo'):
            self.tokenizer = None
        else:
            self.tokenizer = getattr(pytorch_pretrained_bert, self.tokenizer_type).from_pretrained(pre_train,
                                                                                               **tokenizer_args,
                                                                                               cache_dir='bert_cache')
        self.supervised = supervised
        
        df = pandas.read_csv(csv_file)
        self.X = df['text'].tolist()
        self.y = df['label'].tolist() if supervised else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.remove_punc:
            X = re.sub(r'[^\w\s]','',X)

        if self.tokenizer_type.startswith('GPT2'):
            X = self.GPT2_tokenize(X)
        elif self.tokenizer_type.startswith('Basic'):
            X = X.split()
            X = batch_to_ids(X)[0]
        else:
            X = self.tokenize(X)

        if self.supervised:
            y = torch.tensor(self.y[idx])
            y = y-1
        else:
            y = torch.tensor([-1])

        return X, y
    
    def tokenize(self, text):
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        # tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).view(len(indexed_tokens), 1)
        return tokens_tensor

    def GPT2_tokenize(self, text):
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens]).view(len(indexed_tokens), 1)
        return tokens_tensor

    def collect_fn(self, batch):
        return (
            pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0).squeeze(),
            torch.tensor([x[1] for x in batch]),
            MyDataset.make_mask([x[0] for x in batch]),
            MyDataset.make_type_ids([x[0] for x in batch])
        )
    
    @staticmethod
    def make_mask(seqs):
        max_len = max(list(map(len, seqs)))
        mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in seqs]
        mask = torch.LongTensor(mask)
        return mask

    @staticmethod
    def make_type_ids(seqs):
        max_len = max(list(map(len, seqs)))
        type_ids = [[0] * max_len for seq in seqs]
        type_ids = torch.LongTensor(type_ids)
        return type_ids


def get_loader(config):
    dataset_args = Box({
        'train': dict(csv_file=config.data.train.csv_file, 
                      supervised=True, 
                      pre_train=config.embedder.pre_train,
                      remove_punc=config.data.remove_punc if hasattr(config.data, 'remove_punc') else False
                      ),
        'val': dict(csv_file=config.data.dev.csv_file, 
                    supervised=True, 
                    pre_train=config.embedder.pre_train,
                    remove_punc=config.data.remove_punc if hasattr(config.data, 'remove_punc') else False
                    )
    })
    loader_args = Box({
        'train': dict(batch_size=config.train.batch_size, shuffle=True, num_workers=config.num_cpu),
        'val': dict(batch_size=config.val.batch_size, num_workers=config.num_cpu)
    })
    logging.info("Loading dataset...")
    dataset = Box({
        'train': MyDataset(**dataset_args.train),
        'val': MyDataset(**dataset_args.val)
    })
    logging.info("Constructing data loader...")
    loader = Box({
        'train': DataLoader(dataset.train, collate_fn=dataset.train.collect_fn, pin_memory=True, **loader_args.train),
        'val': DataLoader(dataset.val, collate_fn=dataset.val.collect_fn, pin_memory=True, **loader_args.val),
    })
    return loader


def get_pred_loader(csv, config):
    dataset = MyDataset(csv_file=csv, supervised=False, pre_train=config.embedder.pre_train)
    loader = DataLoader(dataset, collate_fn=dataset.collect_fn,
                        pin_memory=True, batch_size=config.val.batch_size, num_workers=config.num_cpu)
    return loader
