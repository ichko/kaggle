import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import src.nn_utils as ut

CHANNEL_DIM = 2


class HyperNCA(ut.Module):
    def __init__(self, feature_size, in_channels):
        super().__init__()

        # TODO: Write transformer indexing soft kernels Conv2D
        self.address_size = 16
        self.in_channels = in_channels

        self.middle_channels = 128
        self.all_in_channels = 16
        self.latent_channels = self.all_in_channels - in_channels

        # +1 for dimensions picking the biases
        self.addresser_1 = ut.LinearAddresser(
            feature_size,
            out_shape=(self.middle_channels, self.all_in_channels + 1),
            address_size=self.address_size,
        )
        self.addresser_2 = ut.LinearAddresser(
            feature_size,
            out_shape=(self.all_in_channels, self.middle_channels + 1),
            address_size=self.address_size,
        )
        self.hyper_conv = ut.HyperConvFilter2D(
            num_filters=1024,
            address_size=self.address_size,
            ks=3,
        )

        self.bn_1 = nn.BatchNorm2d(self.middle_channels)

    def forward(self, task_features, infer_inputs, num_iters):
        # TODO: This should be changed to something more expressive.
        # Now we just avg across demonstrations.
        seq_size = infer_inputs.size(1)
        task_features = torch.mean(task_features, dim=1)

        addresses_1 = self.addresser_1(task_features)
        addresses_2 = self.addresser_2(task_features)
        conv_1 = self.hyper_conv(addresses_1, s=1, p=1, seq_size=seq_size)
        conv_2 = self.hyper_conv(addresses_2, s=1, p=1, seq_size=seq_size)

        def solve_task(x):
            seq_shape = list(x.shape)
            seq_shape.insert(1, num_iters)
            seq = torch.zeros(seq_shape).to(x.device)

            for i in range(num_iters):
                x = conv_1(x)
                # x = F.dropout(x, p=0.01, training=self.training)
                x = self.bn_1(x)
                x = F.leaky_relu(x, negative_slope=0.5)
                x = conv_2(x)

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
        new_shape[CHANNEL_DIM] += self.latent_channels
        new_input = torch.zeros(*new_shape).to(infer_inputs.device)
        new_input[:, :, :self.in_channels] = infer_inputs

        return ut.time_distribute(solve_task, new_input)
