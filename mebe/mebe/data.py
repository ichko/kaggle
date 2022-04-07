import numpy as np
import torch
import pytorch_lightning as pl
import random


class SequencesDataset(torch.utils.data.Dataset):
    TRAIN = np.load('./data/user_train.npy',
                    allow_pickle=True).item()

    TEST = np.load('./data/submission_data.npy',
                   allow_pickle=True).item()

    def __init__(self, mode):
        super().__init__()
        if mode == 'train' or mode == 'val':
            data_json = self.TRAIN
        else:
            data_json = self.TEST

        self.data = [(n, s['keypoints'])
                     for n, s in data_json['sequences'].items()]

        if mode == 'val' or mode == 'train':
            random.seed(1337)
            random.shuffle(self.data)

        val_ratio = 0.15
        train_val_split_pivot = int(len(self.data) * val_ratio)

        if mode == 'train':
            self.data = self.data[train_val_split_pivot:]
        if mode == 'val':
            self.data = self.data[:train_val_split_pivot]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, seq = self.data[idx]
        return name, seq


class SequencesDataModule(pl.LightningDataModule):
    def __init__(self, bs):
        self.bs = bs

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            SequencesDataset(mode='train'), batch_size=self.bs, shuffle=False, num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            SequencesDataset(mode='val'), batch_size=self.bs, shuffle=False, num_workers=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            SequencesDataset(mode='test'), batch_size=self.bs, shuffle=False, num_workers=8,
        )


if __name__ == '__main__':
    dm = SequencesDataModule(bs=64)
    it = iter(dm.train_dataloader())
    batch = next(it)
    print(batch)
