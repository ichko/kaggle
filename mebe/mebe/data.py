import numpy as np
import torch
import pytorch_lightning as pl


class MaskedSequencesDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        if is_train:
            data_json = np.load('data/user_train.npy',
                                allow_pickle=True).item()
        else:
            data_json = np.load('data/submission_data.npy',
                                allow_pickle=True).item()

        self.data = [(n, s['keypoints'])
                     for n, s in data_json['sequences'].items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, seq = self.data[idx]
        return name, seq


class MaskedSequencesDataModule(pl.LightningDataModule):
    def __init__(self, bs):
        self.bs = bs

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            MaskedSequencesDataset(is_train=True), batch_size=self.bs, shuffle=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            MaskedSequencesDataset(is_train=True), batch_size=self.bs, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            MaskedSequencesDataset(is_train=False), batch_size=self.bs, shuffle=False
        )


if __name__ == '__main__':
    dm = MaskedSequencesDataModule(bs=64)
    it = iter(dm.train_dataloader())
    batch = next(it)
    print(batch)
