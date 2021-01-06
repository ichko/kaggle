import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import src.nn_utils as ut

device = 'cpu'

# SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>
# class CAModel(nn.Module):

CHANNEL_DIM = 2


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

    def infer_params(self, task_features, infer_inputs):
        def forward_single_task(features):
            bs = features.size(0)

            batched_conv_w = self.weights.unsqueeze(0)
            batched_conv_w = batched_conv_w.expand(bs, *self.weights.shape)
            batched_conv_b = self.bias.unsqueeze(0)
            batched_conv_b = batched_conv_b.expand(bs, *self.bias.shape)

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
        w = torch.mean(w, dim=1)
        b = torch.mean(b, dim=1)

        num_inference_pairs = infer_inputs.size(1)
        w = ut.unsqueeze_expand(w, dim=1, times=num_inference_pairs)
        b = ut.unsqueeze_expand(b, dim=1, times=num_inference_pairs)

        w = ut.reshape_in_time(w)
        b = ut.reshape_in_time(b)

        self.layer_params = w, b

    def forward(self, x):
        if self.layer_params is None:
            raise Exception('Params are not yet inferred!')

        w, b = self.layer_params
        return ut.batch_conv(x, w, b, p=1)


class CA(nn.Module):
    def __init__(self, num_hyper_kernels, input_channels):
        super().__init__()
        self.conv_1 = HyperConv2D(
            (num_hyper_kernels, 64, input_channels, 3, 3))
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = HyperConv2D(
            (num_hyper_kernels, input_channels, 64, 3, 3))

    def forward(self, task_features, infer_inputs, num_iters):
        self.conv_1.infer_params(task_features, infer_inputs)
        self.conv_2.infer_params(task_features, infer_inputs)

        def solve_task(x):
            seq = []
            for _i in range(num_iters):
                # TODO: Do inference in latent space
                x = self.conv_1(x)
                x = self.bn_1(x)
                x = F.leaky_relu(x, negative_slope=0.5)
                x = self.conv_2(x)
                x = torch.softmax(x, dim=1)
                seq.append(x.clone().unsqueeze(1))

            return torch.log(torch.cat(seq, dim=1))

        return ut.time_distribute(solve_task, infer_inputs)


class HyperRecurrentCNN(ut.Module):
    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

    def __init__(self, input_channels, num_iters):
        super().__init__()
        num_hyper_kernels = 64
        self.num_iters = num_iters

        self.task_feature_extract = ut.time_distribute(nn.Sequential(
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
        ))

        self.ca = CA(
            num_hyper_kernels=num_hyper_kernels,
            input_channels=input_channels,
        )

    def forward(self, batch):
        train_inputs = ut.one_hot(batch['train_inputs'], 11, dim=2)
        train_outputs = ut.one_hot(batch['train_outputs'], 11, dim=2)
        infer_inputs = ut.one_hot(batch['test_inputs'], 11, dim=2)
        train_io = torch.cat([train_inputs, train_outputs], dim=CHANNEL_DIM)
        preds = self.forward_prepared(train_io, infer_inputs)

        return preds[:, :, -1].argmax(dim=CHANNEL_DIM)

    def forward_prepared(self, train_io, infer_inputs):
        task_features = self.task_feature_extract(train_io)
        result = self.ca(task_features, infer_inputs, self.num_iters)
        self.task_features = task_features  # Save to return as info param

        return result

    def criterion_(self, y_pred, y):
        bs, seq = y_pred.shape[:2]
        loss = 0

        seq_dims = list(range(3, y_pred.size(2)))
        weights_sum = 0
        for i in seq_dims:
            weight = (i - seq_dims[0]) / (len(seq_dims) - seq_dims[0] - 1)
            weight = weight**2
            weights_sum += weight

            loss += F.nll_loss(
                input=y_pred[:, :, i].reshape(bs * seq, *y_pred.shape[-3:]),
                target=y.reshape(bs * seq, *y.shape[-2:]),
            ) * weight

        loss /= weights_sum

        return loss

    def optim_step(self, batch, optim_kw={}):
        # TODO: Mask train pairs after using them for inference

        X, y = batch
        max_train = 3
        max_test = 2

        lens = X['all_len']
        all_in = ut.one_hot(X['all_inputs'], num_classes=11, dim=CHANNEL_DIM)
        all_out = ut.one_hot(X['all_outputs'], num_classes=11, dim=CHANNEL_DIM)
        pairs = torch.cat([all_in, all_out], dim=CHANNEL_DIM)

        train, _train_len = ut.sample_padded_sequences(pairs, lens, max_train)
        test, test_len = ut.sample_padded_sequences(pairs, lens, max_test)

        test_in, test_out = test.chunk(2, dim=CHANNEL_DIM)

        y_argmax = torch.argmax(test_out, dim=CHANNEL_DIM)

        y_pred = self.forward_prepared(train, test_in)
        y_pred = ut.mask_seq_from_lens(y_pred, test_len)
        loss = self.criterion_(y_pred, y_argmax)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item(), {
            'X': X,
            'y': y,
            'test_len': test_len,
            'test_inputs': test_in.argmax(dim=CHANNEL_DIM),
            'test_outputs': y_argmax,
            'test_preds': y_pred[:, :, -1].argmax(dim=CHANNEL_DIM),
        }


def make_model(hparams):
    return HyperRecurrentCNN(
        input_channels=hparams['input_channels'],
        num_iters=hparams['nca_iterations'],
    )


def sanity_check():
    pass
