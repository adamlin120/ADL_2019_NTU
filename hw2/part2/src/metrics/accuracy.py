import torch


class Accuracy:
    def __init__(self):
        self.n_corrects = 0.0
        self.n_observed = 0.0
        
    def __len__(self):
        return self.n_observed
    
    def __name__(self):
        return "Accuracy"
    
    def __str__(self):
        return f"{self():.4f}"
    
    def __call__(self):
        return self.n_corrects / self.n_observed
        
    def reset(self):
        self.n_corrects = 0
        self.n_observed = 0
        
    def update(self, pred, y):
        """
        pred: torch.Tensor [num_example x num_class]
        y: torch.Tensor [num_example, ] (value in [0, num_class-1])
        """
        
        self.n_corrects += torch.sum(torch.argmax(pred, 1) == y).item()
        self.n_observed += len(y)