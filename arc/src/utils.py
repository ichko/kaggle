import sys
import os
import torch

# torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def memoize(f):
    memo = {}

    def helper(*args, **kwargs):
        key = str([[str(a) for a in args], kwargs])
        if key not in memo:
            memo[key] = f(*args, **kwargs)

        return memo[key]

    return helper
