import numpy as np
import torch
from tqdm import tqdm

import src.data.preprocess as preprocess


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
class ArcEval:
    def __init__(self):
        self.error = 0
        self.length = 0
        self.num_solved = 0

    def push(self, y_hat_batch, y_batch):
        # Currently outputting single prediction per test input
        assert y_hat_batch.shape == y_batch.shape

        self.length += len(y_batch)
        for y, y_hat in zip(y_batch, y_hat_batch):
            equal = torch.all(y_hat.int() == y.int()).item()
            self.error += not equal
            self.num_solved += equal

    def reduce(self):
        return self.error / self.length, self.num_solved
