import torch
import torch.nn as nn
import ipdb
import pytorch_pretrained_bert

name2BERT_CLASS = {'bert-base-uncased': 'BertModel',
                   'bert-large-uncased': 'BertModel',
                   'bert-base-cased': 'BertModel',
                   'bert-large-cased': 'BertModel',
                   'bert-base-multilingual-uncased': 'BertModel',
                   'bert-base-multilingual-cased': 'BertModel',
                   'bert-base-chinese': 'BertModel',
                   'openai-gpt': 'OpenAIGPTModel',
                   'transfo-xl-wt103': 'TransfoXLModel',
                   'gpt2': 'GPT2Model'
                   }

name2DIM_LAYER = {'bert-base-uncased': 768,
                  'bert-large-uncased': 1024,
                  'bert-base-cased': 768,
                  'bert-large-cased': 1024,
                  'bert-base-multilingual-uncased': 768,
                  'bert-base-multilingual-cased': 768,
                  'bert-base-chinese': 768,
                  'openai-gpt': 768,
                  'transfo-xl-wt103': 1024,
                  'gpt2': 768,
                  'elmo': 1024
                  }




class Embedder(nn.Module):
    def __init__(self, n_bert_layer, pre_train, state_dict=None):
        super(Embedder, self).__init__()
        self.pre_train = pre_train
        self.n_bert_layer = n_bert_layer
        self.dim_layer = name2DIM_LAYER[pre_train]
        self.bert_embed_dim = self.dim_layer * n_bert_layer
        
        self.model = getattr(pytorch_pretrained_bert, name2BERT_CLASS[pre_train]).from_pretrained(pre_train,
                                                                                                  cache_dir='./bert_cache/')
                

    def __call__(self, X, attention_mask=None, token_type_ids=None):
        if self.pre_train.startswith('gpt2'):
            encoded_layers, _ = self.model(X)
            return encoded_layers
        elif self.pre_train.startswith('transfo'):
            encoded_layers, _ = self.model(X)
            return encoded_layers
        elif self.pre_train.startswith('open'):
            encoded_layers = self.model(X)
            return encoded_layers
        elif self.pre_train.startswith('elmo'):
            encoded_layers = torch.stack(self.model(X)['elmo_representations'], 2)
            encoded_layers = encoded_layers.contiguous().view(encoded_layers.size(0), encoded_layers.size(1), -1)
            return encoded_layers

        encoded_layers, _ = self.model(X, attention_mask=attention_mask, token_type_ids=token_type_ids)\
                            if attention_mask is not None else self.model(X)
        encoded_layers = torch.stack(encoded_layers[-self.n_bert_layer:]).permute(1, 2, 0, 3).contiguous()
        encoded_layers = encoded_layers.contiguous().view(encoded_layers.size(0), encoded_layers.size(1), -1)
        return encoded_layers

    @property
    def output_dim(self):
        if self.pre_train.startswith('gpt2'):
            return self.dim_layer
        return self.dim_layer * self.n_bert_layer

    def to_eval(self):
        self.model.eval()

    def to_train(self):
        self.model.train()

    def tune(self, isTrune):
        if isTrune:
            self.to_train()
        else:
            self.to_eval()
