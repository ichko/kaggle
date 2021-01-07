import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.nn_utils as ut


class HyperConv2D(nn.Module):
    # TODO: something multiheaded would be nice!

    def __init__(self, num_kernels, i, o, ks, s=1, p=0, a=None):
        super().__init__()
        self.s = s
        self.p = p
        self.a = a

        # Taken from the src code of nn.ConvND
        self.weights = nn.Parameter(torch.Tensor(num_kernels, o, i, ks, ks))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.Tensor(num_kernels, o))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weights.requires_grad = True
        self.bias.requires_grad = True

        self.layer_params = None

    def infer_params(self, batch_of_features, infer_inputs):
        def infer_params(features):
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

        w, b = ut.time_distribute(infer_params, batch_of_features)

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
        out = ut.batch_conv(x, w, b, p=self.p, s=self.s)

        if self.a is not None:
            out = self.a(out)

        return out
