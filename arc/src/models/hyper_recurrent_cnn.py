import torch
import torch.nn as nn
import torch.nn.functional as F

import src.nn_utils as ut
from src.models.common.hyper_nets import HyperNCA

CHANNEL_DIM = 2


class HyperRecurrentCNN(ut.Module):
    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

    def __init__(self, input_channels, num_iters):
        super().__init__()
        features = 32
        self.num_iters = num_iters

        self.task_feature_extract = ut.time_distribute(nn.Sequential(
            ut.conv_block(
                i=input_channels * 2, o=128, ks=5, s=2, p=2, \
                a=ut.leaky(),
            ),
            ut.conv_block(i=128, o=128, ks=5, s=2, p=2, a=ut.leaky()),
            # nn.Dropout(0.1),
            ut.conv_block(i=128, o=128, ks=5, s=2, p=2, a=ut.leaky()),
            # nn.Dropout(0.05),
            ut.conv_block(i=128, o=64, ks=5, s=2, p=2, a=ut.leaky()),
            ut.conv_block(i=64, o=128, ks=2, s=1, p=0, a=ut.leaky()),
            ut.Reshape(-1, 128),
            nn.Linear(128, features),
        ))

        self.ca = HyperNCA(
            feature_size=features,
            in_channels=input_channels,
        )

    def forward(self, batch, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iters

        preds = self.forward_prepared(
            batch['train'],
            batch['test_in'],
            num_iters,
        )

        return preds[:, :, -1].argmax(dim=CHANNEL_DIM)

    def forward_prepared(self, train, test_in, num_iters):
        task_features = self.task_feature_extract(train)
        pred = self.ca(task_features, test_in, num_iters)
        return pred

    def criterion(self, y_pred, y):
        loss = 0
        seq_dims = list(range(y_pred.size(2)))
        weights_sum = 0

        for i in seq_dims:
            weight = (i + 1) / len(seq_dims)
            weight = (weight + 1) / 2
            weights_sum += weight

            batch_losses = F.nll_loss(
                input=y_pred[:, :, i].reshape(-1, *y_pred.shape[-3:]),
                target=y.reshape(-1, *y.shape[-2:]),
                reduction='none',
            ).mean(dim=(1, 2))

            loss += batch_losses.mean() * weight

        loss /= weights_sum
        # loss /= len(seq_dims)

        return loss, batch_losses  # The last batch losses

    def optim_step(self, batch):
        X, y = batch

        y_pred = self.forward_prepared(
            X['train'],
            X['test_in'],
            self.num_iters,
        )
        y_pred = ut.mask_seq_from_lens(y_pred, X['test_len'])

        loss, batch_losses = self.criterion(y_pred, y)

        if self.training:
            # SRC - https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks
            # predict output from output
            # enforces stability after solution is reached
            y_pred_out = self.forward_prepared(
                X['train'],
                X['test_out'],
                num_iters=1,
            )
            y_pred_out = ut.mask_seq_from_lens(y_pred_out, X['test_len'])

            loss_out_to_out, _ = self.criterion(y_pred_out, y)
            loss = (loss + loss_out_to_out) / 2

        if loss.requires_grad:
            self.optim.zero_grad()
            # import efemarai as ef
            # with ef.scan(wait=True):
            loss.backward()
            self.optim.step()

        y_pred_seq = y_pred.argmax(dim=CHANNEL_DIM + 1)
        y_pred_last = y_pred_seq[:, :, -1]

        return loss.item(), {
            'loss': loss,
            'test_len': X['test_len'],
            'test_in': X['test_in'].argmax(dim=CHANNEL_DIM),
            'test_out': y,
            'test_pred': y_pred_last,
            'test_pred_seq': y_pred_seq,
            'batch_losses': batch_losses,
        }


def make_model(hparams):
    return HyperRecurrentCNN(
        input_channels=hparams['input_channels'],
        num_iters=hparams['nca_iterations'],
    )


def sanity_check():
    pass
