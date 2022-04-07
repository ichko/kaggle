import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

TIME_STEP_SIZE = 256


def remove_nan_entities(x):
    ox = x
    bs, seq, entity, features, dim = x.shape
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(bs * entity, seq * features * dim)
    is_real = torch.any(torch.isnan(x), dim=-1) == False
    x = x[is_real]
    new_entity = x.size(0) // bs
    x = x.view(bs, new_entity, seq, features, dim)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(bs, seq, new_entity, features, dim)
    x = x.contiguous()

    return x


class TransformerDenoisingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        hidden_dim = 20  # 256 // 11
        self.embed_size = 11 * hidden_dim

        self.encoder_features = nn.Sequential(
            nn.Linear(48, hidden_dim)
        )

        self.encoder_entity = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=5), num_layers=2)

        self.encoder_time = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=5), num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(64, 48),
        )

    def forward(self, x):
        bs, seq, entity, features, dim = x.shape

        x = x.view(bs * seq * entity, features * dim)
        x = self.encoder_features(x)

        x = x.view(bs * seq, entity, -1)
        x = self.encoder_entity(x)

        x = x.view(bs, seq, entity, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.view(bs * entity, seq, -1)
        x = self.encoder_time(x)
        x = x.view(bs, entity, seq, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.view(bs, seq, entity, -1)

        x = self.head(x)
        x = x.view(bs, seq, entity, features, dim)

        return x

    def embed(self, inp):
        inp[torch.isnan(inp)] = 0
        batches = torch.split(
            inp, split_size_or_sections=TIME_STEP_SIZE, dim=1)
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

    def optim_step(self, batch):
        names, seq = batch

        # seq = remove_nan_entities(seq)
        seq[torch.isnan(seq)] = 0

        mask_prob = 0.2
        mask = torch.rand_like(seq) > mask_prob
        masked_seq = seq * mask

        seq_batches = torch.split(
            seq, split_size_or_sections=TIME_STEP_SIZE, dim=1)
        masked_seq_batches = torch.split(
            masked_seq, split_size_or_sections=TIME_STEP_SIZE, dim=1)

        loss = 0
        for seq_batch, masked_seq_batch in zip(seq_batches, masked_seq_batches):
            out = self(masked_seq_batch)
            loss += F.mse_loss(out, seq_batch)

        loss /= len(seq_batches)

        return loss

    def training_step(self, batch, batch_idx):
        names, seq = batch
        loss = self.optim_step(batch)
        self.log("train_loss", loss, batch_size=len(seq))
        return loss

    def validation_step(self, batch, batch_idx):
        names, seq = batch
        loss = self.optim_step(batch)
        self.log("val_loss", loss, batch_size=len(seq))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
