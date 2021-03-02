import pytorch_lightning as pl
import torch


class IoU(pl.metrics.Metric):
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.add_state("inter", default=torch.zeros([21]), dist_reduce_fx='sum')
        self.add_state("union", default=torch.zeros([21]), dist_reduce_fx='sum')

    def update(self, preds, target):
        res = preds.argmax(dim=1)
        for index in range(self.n_classes):
            truth = (target.cpu() == index)
            preds = (res == index)

            inter = truth.logical_and(preds.cpu())
            union = truth.logical_or(preds.cpu())

            self.inter[index] += inter.float().sum()
            self.union[index] += union.float().sum()

    def compute(self):
        return self.inter.sum() / self.union.sum()


