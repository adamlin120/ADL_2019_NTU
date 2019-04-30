import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnBaselineNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 dim_rnn=128,
                 dim_speaker=50,
                 num_layers=2,
                 dropout_rate=0):
        super(RnnBaselineNet, self).__init__()
        # dim_embeddings += dim_speaker
        self.dim_embeddings = dim_embeddings
        self.dim_rnn = dim_rnn
        self.dim_speaker = dim_speaker
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.rnn_context = nn.GRU(input_size=dim_embeddings, hidden_size=dim_rnn,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout_rate, bidirectional=True)
        self.rnn_option = nn.GRU(input_size=dim_embeddings, hidden_size=dim_rnn,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout_rate, bidirectional=True)
        self.bi_fc = nn.Bilinear(3*dim_rnn, 3*dim_rnn, 1)

    def forward(self, context, context_lens, options, option_lens, speaker):
        # context [B x ML x E]
        # options [B x (N_P+N_F) x ML x E]

        hid_context = self._encode(context, context_lens, 'context')

        logits = []
        options = options.permute(1, 0, 2, 3)
        for i, option in enumerate(options):
            hid_option = self._encode(option, option_lens[:, i], 'option')

            logit = self.bi_fc(hid_context, hid_option)

            logits.append(logit)
        logits = torch.stack(logits, 1).squeeze()
        return logits

    def _encode(self, seqs, lens, type):
        batch_size = seqs.size(0)
        sorted_lens, sortarg = torch.sort(lens, descending=True)
        unsortarg = torch.argsort(sortarg)

#         seqs = seqs.gather(0, sortarg.view(-1, 1, 1).expand(*seqs.size()))
        seqs = seqs[sortarg]
        seqs = pack_padded_sequence(seqs, sorted_lens, batch_first=True)
        if type == 'context':
            output, hidden = self.rnn_context(
                seqs,
                self._init_hidden(batch_size)
            )
        elif type == 'option':
            output, hidden = self.rnn_option(
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
        # output: [batch_size, 1, max_length, dim_rnn]
        output.unsqueeze_(1)
        # output: [batch_size, dim_rnn]
        output = F.avg_pool2d(output, (output.size(2), 1)).squeeze()
        
        # hidden = hidden[:, unsortarg]
        hidden = hidden[-1].squeeze()
        # hidden = hidden[unsortarg]
        hidden = hidden.gather(0, unsortarg.view(-1, 1).expand(*hidden.size()))
        # hidden = hidden.reshape(hidden.size(0), -1)
        
        return torch.cat((output, hidden), 1)


    def _init_hidden(self, batch_size):
        return torch.zeros(2*self.num_layers, batch_size, self.dim_rnn).cuda()
