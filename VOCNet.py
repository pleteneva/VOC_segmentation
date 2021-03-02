import torch
import pytorch_lightning as pl


class VOCNet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_val = loss(pred, batch['seg'])

        acc_val = iou_metric(pred, batch['seg'])
        self.log('IoU/train', acc_val, on_epoch=True)
        self.log('loss/train', loss_val, on_epoch=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_val = loss(pred, batch['seg'])
        acc_val = iou_metric(pred, batch['seg'])

        self.log('loss/valid', loss_val, on_epoch=True)
        self.log('IoU/valid', acc_val, on_epoch=True)

        return loss_val

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min')
        return self.optim

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        epoch = args[0]
        batch_idx = args[1]

        val_accuracy = self.trainer.logged_metrics['IoU/valid']

        if epoch != 0 and batch_idx == 0:
            self.sched.step(val_accuracy)