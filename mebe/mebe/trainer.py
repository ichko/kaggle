import pytorch_lightning as pl
import torch


class PLTrainer(pl.Module):
    def __init__(self, model):
        super(PLTrainer, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def fit(self, train_dl, val_dl):
        trainer = pl.Trainer(gpus=1, max_epochs=3,
                             progress_bar_refresh_rate=20)
        return trainer.fit(self.model, train_dl, val_dl)
