import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from mebe.data import MaskedSequencesDataModule


class TransformerDenoisingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=48, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(48, 48)

    def forward(self, x):
        bs, seq, dim = x.shape
        x = self.encoder(x)
        x = x.reshape(-1, dim)
        x = self.head(x)
        x = x.reshape(bs, seq, -1)
        return x

    def training_step(self, batch, batch_idx):
        names, sequence = batch

        sequence = sequence[0]
        sequence[torch.isnan(sequence)] = 0
        seq_len, num_flies, _, _ = sequence.shape
        sequence = sequence.reshape(seq_len, num_flies, -1)

        mask_prob = 0.1
        mask = torch.rand_like(sequence) > mask_prob
        masked_batch = sequence * mask
        out = self(masked_batch)
        loss = F.mse_loss(out, sequence)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = TransformerDenoisingModel()
    dm = MaskedSequencesDataModule(bs=1)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(
        model=model,
        train_dataloader=dm.train_dataloader()
    )
