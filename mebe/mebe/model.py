import torch.nn as nn


class Model(nn.Module):
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
