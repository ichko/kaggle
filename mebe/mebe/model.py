import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from mebe.data import MaskedSequencesDataModule


class TransformerDenoisingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        hidden_dim = 20  # 256 // 11
        self.embed_size = 11 * hidden_dim

        self.encoder_features = nn.Sequential(
            nn.Linear(48, hidden_dim)
        )

        self.encoder_entity = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2)
        self.encoder_time = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.Linear(48, 48),
        )

    def forward(self, x):
        bs, seq, entity, features, dim = x.shape
        x = x.view(bs * seq * entity, features * dim)
        x = self.encoder_features(x)
        x = x.view(bs * seq, entity, -1)
        x = self.encoder_entity(x)
        x = self.head(x)
        x = x.view(bs, seq, entity, features, dim)
        return x

    def embed(self, inp):
        inp[torch.isnan(inp)] = 0
        batches = torch.chunk(inp, chunks=10, dim=1)
        outs = []
        for x in batches:
            bs, seq, entity, features, dim = x.shape
            x = x.view(bs * seq * entity, features * dim)
            x = self.encoder_features(x)
            x = x.view(bs * seq, entity, -1)
            x = self.encoder_entity(x)
            x = x.view(bs, seq, -1)

            outs.append(x)

        outs = torch.cat(outs, dim=1)
        return outs

    def training_step(self, batch, batch_idx):
        names, sequence = batch

        sequence = sequence[:, :1000]
        sequence[torch.isnan(sequence)] = 0

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
