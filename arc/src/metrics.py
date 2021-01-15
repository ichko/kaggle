import numpy as np
import torch
from tqdm import tqdm


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
def arc_eval(model, dataloader, num_iters):
    error = 0
    length = 0
    num_solved = 0

    for X, y_batch in tqdm(dataloader):
        # Currently outputting single prediction per test input
        y_hat_batch = model(X, num_iters)
        length += len(y_batch)

        assert y_hat_batch.shape == y_batch.shape

        for y, y_hat in zip(y_batch, y_hat_batch):

            equal = torch.all(y_hat.int() == y.int()).item()
            num_solved += equal
            error += not equal

    return error / length, num_solved


def loss(model, dataloader):
    losses = []
    for batch in dataloader:
        with torch.no_grad():
            loss, _ = model.optim_step(batch)
            losses.append(loss)

    return np.array(losses).mean()