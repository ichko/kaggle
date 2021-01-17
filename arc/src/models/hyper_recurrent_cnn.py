import torch
import torch.nn as nn
import torch.nn.functional as F

import src.nn_utils as ut
from src.models.soft_layer_hyper_conv import SoftLayerConv2D
from src.models.soft_kernel_hyper_conv import SoftKernelConv2D

CHANNEL_DIM = 2


class CA(nn.Module):
    def __init__(self, features, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.latent_size = 21
        # TODO: Write transformer indexing soft kernels Conv2D
        self.conv_1 = SoftKernelConv2D( \
            features_size=features, num_kernels=128,
            i=in_channels + self.latent_size, o=64, ks=3, p=1,
        )
        self.conv_2 = SoftKernelConv2D( \
            features_size=features, num_kernels=128,
            i=64, o=in_channels + self.latent_size, ks=3, p=1,
        )
        # self.conv_1 = \
        #     SoftLayerConv2D(num_kernels, i=num_hidden, o=64, ks=3, p=1)
        # self.conv_2 = \
        #     SoftLayerConv2D(num_kernels, i=64, o=num_hidden, ks=3, p=1)

        self.bn_1 = nn.BatchNorm2d(64)

    def forward(self, task_features, infer_inputs, num_iters):
        self.conv_1.infer_params(task_features, infer_inputs)
        self.conv_2.infer_params(task_features, infer_inputs)

        def solve_task(x):
            seq_shape = list(x.shape)
            seq_shape.insert(1, num_iters)
            seq = torch.zeros(seq_shape).to(x.device)

            for i in range(num_iters):
                x = self.conv_1(x)
                # x = F.dropout(x, p=0.01, training=self.training)
                x = self.bn_1(x)
                x = F.leaky_relu(x, negative_slope=0.5)
                x = self.conv_2(x)

                # Softmax only the input channels
                x[:, :self.in_channels] = \
                    torch.softmax(x[:, :self.in_channels].clone(), dim=1)
                x[:, self.in_channels:] = \
                    torch.tanh(x[:, self.in_channels:].clone())

                # x = torch.softmax(x, dim=1)
                seq[:, i] = x

            seq = seq[:, :, :self.in_channels]
            return torch.log(seq)

        new_shape = list(infer_inputs.shape)
        new_shape[CHANNEL_DIM] += self.latent_size
        new_input = torch.ones(*new_shape).to(infer_inputs.device)
        new_input[:, :, :self.in_channels] = infer_inputs

        return ut.time_distribute(solve_task, new_input)


class HyperRecurrentCNN(ut.Module):
    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

    def __init__(self, input_channels, num_iters):
        super().__init__()
        features = 128
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

        self.ca = CA(
            features=features,
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

    def forward_prepared(self, train_io, infer_inputs, num_iters):
        task_features = self.task_feature_extract(train_io)
        result = self.ca(task_features, infer_inputs, num_iters)
        self.task_features = task_features  # Save to return as info param

        return result

    def criterion(self, y_pred, y):
        loss = 0
        seq_dims = list(range(y_pred.size(2)))
        weights_sum = 0

        for i in seq_dims:
            weight = (i + 1) / len(seq_dims)
            weight = (weight + 1) / 2
            weights_sum += weight

            loss += F.nll_loss(
                input=y_pred[:, :, i].reshape(-1, *y_pred.shape[-3:]),
                target=y.reshape(-1, *y.shape[-2:]),
            )
            loss *= weight

        # loss /= weights_sum
        loss /= len(seq_dims)

        return loss

    def optim_step(self, batch):
        X, y = batch

        y_pred = self.forward_prepared(
            X['train'],
            X['test_in'],
            self.num_iters,
        )
        y_pred = ut.mask_seq_from_lens(y_pred, X['test_len'])

        loss = self.criterion(y_pred, y)

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

            loss_out_to_out = self.criterion(y_pred_out, y)
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
        }


def make_model(hparams):
    return HyperRecurrentCNN(
        input_channels=hparams['input_channels'],
        num_iters=hparams['nca_iterations'],
    )


def sanity_check():
    pass
