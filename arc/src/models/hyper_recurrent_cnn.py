import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import src.nn_utils as ut

device = 'cpu'

# SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>
# class CAModel(nn.Module):


class HyperCNN(nn.Module):
    # TODO: something multiheaded would be nice!

    def __init__(self, shape):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(shape))
        # TODO: Add bias parameter
        # self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.weights.requires_grad = True
        self.layer_params = None

    def forward_single_task(self, task_features):
        repeated_dims = [
            task_features.shape[0], *([1] * len(self.weights.shape))
        ]
        batched_conv_params = self.weights.unsqueeze(0).repeat(*repeated_dims)

        task_features = task_features.unsqueeze(2).unsqueeze(2).unsqueeze(
            2).unsqueeze(2)

        weighted_conv_params = task_features * batched_conv_params
        conv_params_per_demonstration = torch.sum(
            weighted_conv_params,
            dim=1,
        )

        return conv_params_per_demonstration

    def infer_params(self, task_features):
        conv_params_per_demonstration = ut.time_distribute(
            self.forward_single_task, task_features)

        # TODO: This should be changed to something more expressive.
        # Now we just avg across demonstrations.
        avg_across_demonstrations = torch.mean(
            conv_params_per_demonstration,
            dim=1,
        )

        self.layer_params = avg_across_demonstrations

    def forward(self, x):
        if self.layer_params is None:
            raise Exception('Params are not yet inferred!')

        x = ut.batch_conv(x, self.layer_params, p=2)
        return x


class HyperRecurrentCNN(ut.Module):
    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

    def __init__(self, input_channels, num_iters):
        super().__init__()
        num_hyper_kernels = 32
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

        self.conv_params_1 = HyperCNN(
            (num_hyper_kernels, 128, input_channels, 5, 5))
        self.conv_params_2 = HyperCNN(
            (num_hyper_kernels, input_channels, 128, 5, 5))

    def forward(self, batch):
        train_inputs = batch['train_inputs']
        train_outputs = batch['train_outputs']
        test_inputs = batch['test_inputs']

        channel_dim = 2
        train_io = torch.cat([train_inputs, train_outputs], dim=channel_dim)

        task_features = self.task_feature_extract(train_io)
        self.conv_params_1.infer_params(task_features)
        self.conv_params_2.infer_params(task_features)

        def solve_task(x):
            for _i in range(self.num_iters):
                # TODO: Add batch norm
                x = self.conv_params_1(x)
                x = F.relu(x)
                x = self.conv_params_2(x)
                x = torch.softmax(x, dim=1)

            return torch.log(x)

        result = ut.time_distribute(solve_task, test_inputs)
        return result

    def optim_step(self, batch, optim_kw={}):
        X, y = batch

        channel_dim = 2
        y_argmax = torch.argmax(y, dim=channel_dim)
        y_pred = self.optim_forward(X)

        bs, seq = y_pred.shape[:2]
        loss = F.nll_loss(
            input=y_pred.reshape(bs * seq, *y_pred.shape[-3:]),
            target=y_argmax.reshape(bs * seq, *y_argmax.shape[-2:]),
        )

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item(), {
            'X': X,
            'y_pred': y_pred,
            'y': y,
        }


def make_model(hparams):
    return HyperRecurrentCNN(
        input_channels=hparams['input_channels'],
        num_iters=hparams['nca_iterations'],
    )


def sanity_check():
    pass
