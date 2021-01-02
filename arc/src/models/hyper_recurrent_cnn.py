import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import src.nn_utils as ut

device = 'cpu'

# SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>
# class CAModel(nn.Module):


class HyperConv2D(nn.Module):
    # TODO: something multiheaded would be nice!

    def __init__(self, shape):
        super().__init__()

        self.weights = nn.Parameter(torch.Tensor(*shape))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Taken from the src code of nn.ConvND
        self.bias = nn.Parameter(torch.Tensor(*shape[:2]))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weights.requires_grad = True
        self.bias.requires_grad = True

        self.layer_params = None

    def infer_params(self, task_features):
        def forward_single_task(features):
            rep_dims_w = [features.shape[0], *([1] * len(self.weights.shape))]
            rep_dims_b = [features.shape[0], *([1] * len(self.bias.shape))]

            batched_conv_w = self.weights.unsqueeze(0).repeat(*rep_dims_w)
            batched_conv_b = self.bias.unsqueeze(0).repeat(*rep_dims_b)

            features_for_w = features.unsqueeze(2).unsqueeze(2) \
                                     .unsqueeze(2).unsqueeze(2)
            features_for_b = features.unsqueeze(2)

            weighted_conv_w = features_for_w * batched_conv_w
            weighted_conv_b = features_for_b * batched_conv_b

            sum_weighted_w = torch.sum(weighted_conv_w, dim=1)
            sum_weighted_b = torch.sum(weighted_conv_b, dim=1)

            return sum_weighted_w, sum_weighted_b

        w, b = ut.time_distribute(forward_single_task, task_features)

        # TODO: This should be changed to something more expressive.
        # Now we just avg across demonstrations.
        w_mean = torch.mean(w, dim=1)
        b_mean = torch.mean(b, dim=1)

        self.layer_params = w_mean, b_mean

    def forward(self, x):
        if self.layer_params is None:
            raise Exception('Params are not yet inferred!')

        w, b = self.layer_params
        x = ut.batch_conv(x, w, b, p=1)

        return x


class CA(nn.Module):
    def __init__(self, num_hyper_kernels, input_channels):
        super().__init__()
        self.conv_1 = HyperConv2D(
            (num_hyper_kernels, 64, input_channels, 3, 3))
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = HyperConv2D(
            (num_hyper_kernels, input_channels, 64, 3, 3))

    def forward(self, task_features, test_inputs, num_iters):
        self.conv_1.infer_params(task_features)
        self.conv_2.infer_params(task_features)

        def solve_task(x):
            seq = []
            for _i in range(num_iters):
                x = self.conv_1(x)
                x = self.bn_1(x)
                x = F.leaky_relu(x, negative_slope=0.5)
                x = self.conv_2(x)
                x = torch.softmax(x, dim=1)
                seq.append(x.clone().unsqueeze(1))

            return torch.log(torch.cat(seq, dim=1))

        return ut.time_distribute(solve_task, test_inputs)


class HyperRecurrentCNN(ut.Module):
    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

    def __init__(self, input_channels, num_iters):
        super().__init__()
        num_hyper_kernels = 64
        self.num_iters = num_iters

        self.task_feature_extract = nn.Sequential(
            ut.conv_block(
                i=input_channels * 2, o=128, ks=5, s=2, p=2, \
                a=ut.leaky(),
            ),
            ut.conv_block(i=128, o=128, ks=5, s=2, p=2, a=ut.leaky()),
            ut.conv_block(i=128, o=64, ks=5, s=2, p=2, a=ut.leaky()),
            ut.conv_block(i=64, o=64, ks=5, s=2, p=2, a=ut.leaky()),
            ut.conv_block(i=64, o=128, ks=2, s=1, p=0, a=ut.leaky()),
            ut.Reshape(-1, 128),
            nn.Linear(128, num_hyper_kernels),
            nn.Softmax(dim=1),
        )
        self.task_feature_extract = ut.time_distribute(
            self.task_feature_extract)

        self.ca = CA(
            num_hyper_kernels=num_hyper_kernels,
            input_channels=input_channels,
        )

    def forward(self, batch):
        train_inputs = batch['train_inputs']
        train_outputs = batch['train_outputs']
        test_inputs = batch['test_inputs']

        channel_dim = 2
        train_io = torch.cat([train_inputs, train_outputs], dim=channel_dim)

        task_features = self.task_feature_extract(train_io)
        result = self.ca(task_features, test_inputs, self.num_iters)

        return result

    def optim_step(self, batch, optim_kw={}):
        X, y = batch

        channel_dim = 2
        y_argmax = torch.argmax(y, dim=channel_dim)
        y_pred = self.optim_forward(X)

        bs, seq = y_pred.shape[:2]
        loss = 0

        seq_dims = list(range(3, y_pred.size(2)))
        weights_sum = 0
        for i in seq_dims:
            weight = ((i - seq_dims[0]) / (len(seq_dims) - seq_dims[0] - 1))**4
            weights_sum += weight
            loss += F.nll_loss(
                input=y_pred[:, :, i].reshape(bs * seq, *y_pred.shape[-3:]),
                target=y_argmax.reshape(bs * seq, *y_argmax.shape[-2:]),
            ) * weight

        loss /= weights_sum

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item(), {
            'X': X,
            'y_pred': y_pred[:, :, -1],
            'y': y,
        }


def make_model(hparams):
    return HyperRecurrentCNN(
        input_channels=hparams['input_channels'],
        num_iters=hparams['nca_iterations'],
    )


def sanity_check():
    pass
