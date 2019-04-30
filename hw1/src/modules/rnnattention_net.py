import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnAttentionNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 dim_rnn=128,
                 num_layers=2,
                 dropout_rate=0,
                 similarity='inner',
                 pooling='avg'):
        super(RnnAttentionNet, self).__init__()
        self.dim_embeddings = dim_embeddings
        self.dim_rnn = dim_rnn
        self.dim_encoded = 2*dim_rnn
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.similarity = similarity
        self.pooling = pooling

        self.rnn_context = nn.GRU(input_size=dim_embeddings, hidden_size=dim_rnn,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout_rate, bidirectional=True)
#         self.rnn_option = nn.GRU(input_size=dim_embeddings, hidden_size=dim_rnn,
#                           num_layers=num_layers, batch_first=True,
#                           dropout=dropout_rate, bidirectional=True)
        
        self.rnn_attn_context = nn.GRU(input_size=4*self.dim_encoded, hidden_size=dim_rnn,
                          num_layers=1, batch_first=True,
                          dropout=dropout_rate, bidirectional=True)
#         self.rnn_attn_option = nn.GRU(input_size=4*self.dim_encoded, hidden_size=dim_rnn,
#                           num_layers=num_layers, batch_first=True,
#                           dropout=dropout_rate, bidirectional=True)
        if self.similarity == 'cosine' or self.similarity == 'inner':
            self.bi_fc = nn.Bilinear(self.dim_encoded, self.dim_encoded, 1)
        elif self.similarity == 'trilinear':
            self.co_attn = COAttention(d_model=self.dim_encoded, dropout=dropout_rate)
            self.fc_co = nn.Linear(self.dim_encoded, 1)
        else:
            raise ValueError(f"Invalid Similarity: {self.similarity}")            

    def forward(self, context, context_lens, options, option_lens, speaker):
        # context [batch_size x max_context_len x dim_embeddings]
        # options [batch_size x (N_P+N_F) x max_option_len x dim_embeddings]

        hid_context = self._encode(context, context_lens, 'context')
        context_mask = torch.zeros(hid_context.size(0), hid_context.size(1), 
                                   dtype=torch.uint8)
        for i, length in enumerate(context_lens):
            context_mask[i, :length] = 1        

        logits = []
        options = options.permute(1, 0, 2, 3)
        for i, option in enumerate(options):
            option_len = option_lens[:, i]
            hid_option = self._encode(option, option_len, 'option')
            
            option_mask = torch.zeros(hid_option.size(0), hid_option.size(1),
                                      dtype=torch.uint8)
            for j, length in enumerate(option_len):
                option_mask[j, :length] = 1  

            # attn_mat: [batch_size, max_context_len, max_option_lne]
            # attn_context: [batch_size, max_context_len, dim_encode]
            # attn_option: [batch_size, max_option_len, dim_encode]
            if self.similarity == 'cosine' or self.similarity == 'inner':
                if self.similarity == 'cosine':
                    hid_context = hid_context / (torch.norm(hid_context.detach(), dim=2, keepdim=True) + 1e-5)
                    hid_option = hid_option / (torch.norm(hid_option.detach(), dim=2, keepdim=True) + 1e-5)
                    attn_mat = hid_context.bmm(hid_option.transpose(1, 2))
                elif self.similarity == 'inner':
                    attn_mat = hid_context.bmm(hid_option.transpose(1, 2)) / math.sqrt(self.dim_encoded)
                    
                attn_context = F.softmax(attn_mat, dim=2).bmm(hid_option)
                attn_option = F.softmax(attn_mat, dim=1).transpose(2, 1).bmm(hid_context)
                
#                 breakpoint()


                # attn_context: [batch_size, max_context_len, 4*dim_encode]
                # attn_option: [batch_size, max_option_len, 4*dim_encode]
                attn_context = torch.cat(
                    (attn_context, hid_context, attn_context*hid_context, attn_context-hid_context),
                    2)            
                attn_option = torch.cat(
                    (attn_option, hid_option, attn_option*hid_option, attn_option-hid_option),
                    2)

                attn_context = self._encode_attn(attn_context, context_lens, 'context')
                attn_option = self._encode_attn(attn_option, option_lens[:, i], 'option') 

                logit = self.bi_fc(attn_context, attn_option)
            elif self.similarity == 'trilinear':
                attn_context = self.co_attn(hid_context, hid_option, context_mask, option_mask)
                attn_context = self._encode_attn(attn_context, context_lens, 'context')
                logit = self.fc_co(attn_context)
            

            logits.append(logit)
        logits = torch.stack(logits, 1).squeeze()
        return logits

    def _encode(self, seqs, lens, type):
        batch_size = seqs.size(0)
        sorted_lens, sortarg = torch.sort(lens, descending=True)
        unsortarg = torch.argsort(sortarg)

        seqs = seqs[sortarg]
        seqs = pack_padded_sequence(seqs, sorted_lens, batch_first=True)
        if type == 'context':
            output, hidden = self.rnn_context(
                seqs,
                self._init_hidden(batch_size)
            )
        elif type == 'option':
            output, hidden = self.rnn_context(
                seqs,
                self._init_hidden(batch_size)
            )
           
        # output: [batch_size, max_length, 2*dim_rnn]
        output, _ = pad_packed_sequence(output, batch_first=True)
        # unsort
        output = output[unsortarg]
        '''
        # output: [batch_size, max_length, dim_rnn]
        output = (output[:, :, :self.dim_rnn] + output[:, :, self.dim_rnn:]) / 2
        '''

        return output
    
    def _encode_attn(self, attn_seqs, attn_lens, type):
        attn_batch_size = attn_seqs.size(0)
        attn_sorted_lens, attn_sortarg = torch.sort(attn_lens, descending=True)
        attn_unsortarg = torch.argsort(attn_sortarg)

        attn_seqs = attn_seqs[attn_sortarg]
        attn_seqs = pack_padded_sequence(attn_seqs, attn_sorted_lens, batch_first=True)
        if type == 'context':
            attn_output, attn_hidden = self.rnn_attn_context(attn_seqs)
        elif type == 'option':
            attn_output, attn_hidden = self.rnn_attn_context(attn_seqs)
           
        # output: [batch_size, max_length, 2*dim_rnn]
        attn_output, _ = pad_packed_sequence(attn_output, batch_first=True)
        # unsort
        attn_output = attn_output[attn_unsortarg]
        
        # attn_output: [batch_size, 1, max_length, dim_rnn]
        attn_output = attn_output.unsqueeze(1)
        # attn_output: [batch_size, dim_rnn]
        if self.pooling == 'avg':
            attn_output = F.avg_pool2d(attn_output, (attn_output.size(2), 1)).squeeze()
        elif self.pooling == 'max':
            attn_output = F.max_pool2d(attn_output, (attn_output.size(2), 1)).squeeze()

        return attn_output

    def _init_hidden(self, batch_size):
        return torch.zeros(2*self.num_layers, batch_size, self.dim_rnn).cuda()

    
class COAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout
        
    def _masking(self, target, mask):
        mask = mask.type(torch.float32).cuda()
        return target * mask + (1 - mask) * (-1e10)

    def forward(self, C, Q, Cmask, Qmask):
        batch_size, Lc, _ = C.size()
        Lq = Q.size(1)
        Cmask = Cmask.view(batch_size, Lc, 1)
        Qmask = Qmask.view(batch_size, 1, Lq)
        
        S = self.trilinear_attn(C, Q)
        S1 = F.softmax(self._masking(S, Qmask), dim=2)
        S2 = F.softmax(self._masking(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def trilinear_attn(self, C, Q):
        Lc = C.size(1)
        Lq = Q.size(1)
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res