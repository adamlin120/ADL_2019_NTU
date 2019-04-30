import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from .embedder import Embedder
from pytorch_pretrained_bert import BertForSequenceClassification

class FC(nn.Module):
    def __init__(self, num_class, fc_dims, dropout_rate, fine_tune_embedder, correct_last=False,
                 **embedder_args):
        super(FC, self).__init__()
        self.correct_last = correct_last
        self.fine_tune_embedder = fine_tune_embedder
        self.embedder = Embedder(**embedder_args)
        self.embedder.tune(fine_tune_embedder)
        fc_dims = [self.embedder.output_dim] + fc_dims + [num_class]
        self.fc = nn.ModuleList([nn.Linear(fc_dims[i], layers_size) for i, layers_size in enumerate(fc_dims[1:])])
        self.bn_embedder = nn.BatchNorm1d(self.embedder.output_dim)
        self.bn = nn.ModuleList([nn.BatchNorm1d(layers_size) for layers_size in fc_dims[1:]])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        with torch.set_grad_enabled(self.fine_tune_embedder):
            if self.fine_tune_embedder:
                self.embedder.to_train()
            else:
                self.embedder.to_eval()

            x = self.embedder(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
            #for i, mask in enumerate(attention_mask):
            #    seq_len = torch.sum(mask)
            #    x[i, 0, :] = torch.mean(x[i, 1:(seq_len-1)], 0)
            #x = x[:, 0, :]
            x = torch.stack(
                    [torch.mean(x[i, 1:(mask.sum()-1)], 0)
                     for i, mask in enumerate(attention_mask)]
                    )

        x = self.bn_embedder(x)
        x = self.dropout(x)
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if self.correct_last:
                if i != len(self.fc)-1:
                    x = self.bn[i](x)
                    x = F.leaky_relu(x)
                    x = self.dropout(x)
            else:
                x = self.bn[i](x)
                x = F.leaky_relu(x)
                if i!=len(self.fc)-1:
                    x = self.dropout(x)
        return x

    def to_train(self):
        self.train()
        self.embedder.to_train()

    def to_eval(self):
        self.eval()
        self.embedder.to_eval()


class BertFC(nn.Module):
    def __init__(self, num_class, pre_train, fc_dims, dropout_rate, correct_last=False):
        super(BertFC, self).__init__()
        if fc_dims[0]==-1 or len(fc_dims)==0:
            fc_dims = [num_class]
        else:
            fc_dims += [num_class]

        self.correct_last = correct_last

        self.model = BertForSequenceClassification.from_pretrained(pre_train, num_labels=fc_dims[0], cache_dir='./bert_cache/')


        self.fc = nn.ModuleList([nn.Linear(fc_dims[i], dim) for i, dim in enumerate(fc_dims[1:])])
        self.bn_embedder = nn.BatchNorm1d(fc_dims[0])
        self.bn = nn.ModuleList([nn.BatchNorm1d(dim) for dim in fc_dims[1:]])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask, token_type_ids):
        x = self.model(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.bn_embedder(x)
        x = self.dropout(x)
#        for i, fc in enumerate(self.fc):
#            x = fc(x)
#            x = self.bn[i](x)
#            x = F.leaky_relu(x)
#            if i!=len(self.fc)-1:
#                x = self.dropout(x)
#        return x
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if self.correct_last:
                if i != len(self.fc)-1:
                    x = self.bn[i](x)
                    x = F.leaky_relu(x)
                    x = self.dropout(x)
            else:
                x = self.bn[i](x)
                x = F.leaky_relu(x)
                if i!=len(self.fc)-1:
                    x = self.dropout(x)
        return x


    def to_train(self):
        self.train()
        self.model.train()

    def to_eval(self):
        self.eval()
        self.model.eval()

