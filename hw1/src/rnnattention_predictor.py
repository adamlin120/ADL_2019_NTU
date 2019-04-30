import torch
from base_predictor import BasePredictor
from modules.rnnattention_net import RnnAttentionNet


class RnnAttentionPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                 dropout_rate=0.2, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner', dim_rnn=128, num_layers=2, pooling='avg',
                 **kwargs):
        super(RnnAttentionPredictor, self).__init__(**kwargs)
        self.model = RnnAttentionNet(embedding.size(1),
                                    dim_rnn=dim_rnn,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate,
                                    similarity=similarity)
        self.embedding = torch.nn.Embedding.from_pretrained(embedding)

        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(),
        #                                   lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)


        self.loss_name = loss
        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss(),
            'CELoss': torch.nn.CrossEntropyLoss()
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))  # [B x ML x E]
            options = self.embedding(batch['options'].to(self.device))  # [B x (N_P+N_F) x ML x E]


        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'].to(self.device),
            options.to(self.device),
            batch['option_lens'].to(self.device),
            batch['speaker']
        )

        if self.loss_name == 'CELoss':
            labels = torch.ones(len(batch['labels']), dtype=torch.long, device=self.device)
        elif self.loss_name == 'BCELoss':
            labels = batch['labels'].type(torch.FloatTensor).to(self.device)
        loss = self.loss(logits, labels)

        return logits, loss

    def _predict_batch(self, batch):
#         breakpoint()
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'].to(self.device),
            options.to(self.device),
            batch['option_lens'].to(self.device),
            batch['speaker']
        )
        return logits
