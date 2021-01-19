import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import src.nn_utils as ut

CHANNEL_DIM = 2


class LinearAddresser(nn.Module):
    def __init__(self, in_features_size, out_shape, address_size):
        super().__init__()
        self.num_addresses = np.prod(out_shape)
        self.out_shape = out_shape
        self.address_size = address_size

        self.dense = nn.Linear(
            in_features_size,
            address_size * self.num_addresses,
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, *self.out_shape, self.address_size)
        x = torch.softmax(x, dim=-1)
        return x


class HyperConv2D(nn.Module):
    def __init__(self, num_filters, ks):
        super().__init__()

        self.num_filters = num_filters
        self.w_bank = nn.Parameter(torch.Tensor(num_filters, ks, ks))
        nn.init.kaiming_uniform_(self.w_bank, a=math.sqrt(5))

        self.b_bank = nn.Parameter(torch.Tensor(num_filters))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_bank[0])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_bank, -bound, bound)

        self.w_bank.requires_grad = True
        self.b_bank.requires_grad = True

    def forward(self, addresses, s, p, seq_size=0):
        """
            {addresses} - should be tensor with size
                (bs, output_addresses + 1, input_addresses, num_filters)
                output_addresses + 1 for the bias
                The actual tensor bank used to convolve the input is inferred from the banks
        """
        assert addresses.size(-1) == self.num_filters

        w_addresses = addresses[:, :, 1:]
        b_addresses = addresses[:, :, 0]
        w = ut.softmax_address(w_addresses, self.w_bank)
        b = ut.softmax_address(b_addresses, self.b_bank)

        if seq_size > 0:
            w = ut.unsqueeze_expand(w, dim=1, times=seq_size)
            b = ut.unsqueeze_expand(b, dim=1, times=seq_size)

            w = ut.reshape_in_time(w)
            b = ut.reshape_in_time(b)

        return ut.InferredConv2D(w, b, s, p)


class HyperNCA(nn.Module):
    def __init__(self, feature_size, in_channels):
        super().__init__()

        # TODO: Write transformer indexing soft kernels Conv2D
        self.address_size = 128
        self.in_channels = in_channels

        self.middle_channels = 32
        self.all_in_channels = 32
        self.latent_channels = self.all_in_channels - in_channels

        # +1 for dimensions picking the biases
        self.addresser_1 = LinearAddresser(
            feature_size,
            out_shape=(self.middle_channels, self.all_in_channels + 1),
            address_size=self.address_size,
        )
        self.addresser_2 = LinearAddresser(
            feature_size,
            out_shape=(self.all_in_channels, self.middle_channels + 1),
            address_size=self.address_size,
        )
        self.hyper_conv = HyperConv2D(num_filters=self.address_size, ks=3)
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
        new_input = torch.ones(*new_shape).to(infer_inputs.device)
        new_input[:, :, :self.in_channels] = infer_inputs

        return ut.time_distribute(solve_task, new_input)
