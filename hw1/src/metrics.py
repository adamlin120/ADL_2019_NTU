import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def __str__(self):
        return self.name

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        argsort = torch.argsort(predicts, 1, descending=True)
        for i, row in enumerate(batch['labels']):
            for j, element in enumerate(row):
                if (element == 1):
                    if (argsort[i, j] <= (self.at-1)):
                        self.n_correct += 1
                        break
        self.n += len(predicts)

    def get_score(self):
        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
