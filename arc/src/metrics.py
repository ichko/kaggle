import numpy as np
import torch
from tqdm import tqdm

import src.data.preprocess as preprocess


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
def arc_eval(optim_iterable):
    error = 0
    length = 0
    num_solved = 0

    for info in tqdm(optim_iterable):
        # Currently outputting single prediction per test input
        y_hat_batch = info['y_pred']
        y_batch = info['y']

        length += len(y_batch)

        assert y_hat_batch.shape == y_batch.shape

        for y, y_hat in zip(y_batch, y_hat_batch):
            equal = torch.all(y_hat.int() == y.int()).item()
            num_solved += equal
            error += not equal

    return error / length, num_solved
