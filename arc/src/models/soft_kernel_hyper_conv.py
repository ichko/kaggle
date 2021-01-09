import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.nn_utils as ut


class SoftKernelConv2D(nn.Module):
    def __init__(self, features_size, num_kernels, i, o, ks, s=1, p=0, a=None):
        super().__init__()
        self.s = s
        self.p = p
        self.a = a
        self.i = i
        self.o = o
        self.features_size = features_size
        self.num_kernels = num_kernels

        # Taken from the src code of nn.ConvND
        self.w_bank = nn.Parameter(torch.Tensor(num_kernels, i, ks, ks))
        nn.init.kaiming_uniform_(self.w_bank, a=math.sqrt(5))

        self.b_bank = nn.Parameter(torch.Tensor(num_kernels, o))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_bank[0])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_bank, -bound, bound)

        # +1 for dimensions picking the biases
        self.embed = nn.Linear(self.features_size, (o + 1) * num_kernels)

        self.w_bank.requires_grad = True
        self.b_bank.requires_grad = True

        self.layer_params = None

    def infer_params(self, batch_of_features, infer_inputs):
        def infer_params(features):
            bs = features.size(0)

            w = ut.unsqueeze_expand(self.w_bank, dim=0, times=bs * self.o)
            b = ut.unsqueeze_expand(self.b_bank, dim=0, times=bs)

            pickers = self.embed(features)

            pickers_w = pickers[:, self.num_kernels:]
            pickers_b = pickers[:, :self.num_kernels]
            pickers_w = pickers_w.reshape(bs * self.o, self.num_kernels)

            pickers_w = torch.softmax(pickers_w, dim=1)
            pickers_b = torch.softmax(pickers_b, dim=1)

            pickers_w = ut.unsqueeze_like(pickers_w, w)
            pickers_b = ut.unsqueeze_like(pickers_b, b)

            soft_w = torch.sum(pickers_w * w, dim=1)
            soft_b = torch.sum(pickers_b * b, dim=1)

            soft_w = soft_w.view(bs, self.o, *soft_w.shape[1:])

            return soft_w, soft_b

        w, b = ut.time_distribute(infer_params, batch_of_features)

        # TODO: This should be changed to something more expressive.
        # Now we just avg across demonstrations.
        # TODO: AVG THE FEATURES INSTEAD OF THE WEIGHTS
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
        out = ut.batch_conv(x, w, b, p=self.p, s=self.s)

        if self.a is not None:
            out = self.a(out)

        return out
