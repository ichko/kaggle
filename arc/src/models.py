import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import nn_utils as ut

from tqdm.auto import tqdm

device = 'cpu'


class CAModel(nn.Module):
    # SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>

    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1))

    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x


class HyperParams(nn.Module):
    # TODO: something multiheaded would be nice!

    def __init__(self, shape):
        super().__init__()
        self.params = nn.Parameter(torch.randn(shape))
        self.params.requires_grad = True

    def forward_single_task(self, task_features):
        repeated_dims = [
            task_features.shape[0], *([1] * len(self.params.shape))
        ]
        batched_conv_params = self.params.unsqueeze(0).repeat(*repeated_dims)

        task_features = task_features.unsqueeze(2).unsqueeze(2).unsqueeze(
            2).unsqueeze(2)

        weighted_conv_params = task_features * batched_conv_params
        conv_params_per_demonstration = torch.sum(
            weighted_conv_params,
            dim=1,
        )

        return conv_params_per_demonstration

    def forward(self, task_features):
        conv_params_per_demonstration = ut.time_distribute(
            self.forward_single_task, task_features)

        # TODO: This should be changed to something more expressive.
        # Now we just avg across demonstrations.
        avg_across_demonstrations = torch.mean(
            conv_params_per_demonstration,
            dim=1,
        )

        return avg_across_demonstrations


class SoftAddressableComputationCNN(ut.Module):
    def __init__(self, input_channels):
        super().__init__()

        num_hyper_kernels = 32

        self.task_feature_extract = nn.Sequential(
            nn.Conv2d(
                input_channels * 2,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            ut.Reshape(-1, 128),
            nn.Linear(128, num_hyper_kernels),
            nn.Softmax(dim=1),
        )
        self.task_feature_extract = ut.time_distribute(
            self.task_feature_extract)

        self.conv_params_1 = HyperParams(
            (num_hyper_kernels, 128, input_channels, 5, 5))
        self.conv_params_2 = HyperParams(
            (num_hyper_kernels, input_channels, 128, 5, 5))

    def forward(self, batch):
        train_inputs = batch['train_inputs']
        train_outputs = batch['train_outputs']
        test_inputs = batch['test_inputs']

        channel_dim = 2
        train_io = torch.cat([train_inputs, train_outputs], dim=channel_dim)

        task_features = self.task_feature_extract(train_io)
        layer_1 = self.conv_params_1(task_features)
        layer_2 = self.conv_params_2(task_features)

        # TODO: This assumes only single test pair
        test_inputs = test_inputs[:, 0]
        x = test_inputs
        for i in range(10):
            x = ut.batch_conv(x, layer_1, p=2)
            x = F.relu(x)
            x = ut.batch_conv(x, layer_2, p=2)
            x = torch.softmax(x, dim=1)

        return x.unsqueeze(1)


def solve_task(task, max_steps=10):
    model = CAModel(11).to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=(0.1 / (num_steps * 2)))

        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for sample in task:
                # predict output from input
                x = torch.from_numpy(inp2img(
                    sample["input"])).unsqueeze(0).float().to(device)
                y = torch.tensor(
                    sample["output"]).long().unsqueeze(0).to(device)
                y_pred = model(x, num_steps)
                loss += criterion(y_pred, y)

                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(inp2img(
                    sample["output"])).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1)
                loss += criterion(y_pred, y)

            loss.backward()
            print(loss.item())
            optimizer.step()
            losses[(num_steps - 1) * num_epochs + e] = loss.item()
    return model, num_steps, losses


@torch.no_grad()
def predict(model, task):
    predictions = []
    for sample in task:
        x = torch.from_numpy(inp2img(
            sample["input"])).unsqueeze(0).float().to(device)
        pred = model(x, 100).argmax(1).squeeze().cpu().numpy()
        predictions.append(pred)
    return predictions


def train(model, dataloader, epochs):
    for epoch in tqdm(range(epochs)):
        tq = tqdm(dataloader)
        for batch in tq:
            loss, info = model.optim_step(batch)

            tq.set_description(f'LOSS: {loss:.5f}')


# https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
# AVG TOP 3 for each task (less is better)
def evaluate(model, dataloader):
    error = 0
    for X, y in tqdm(dataloader):
        # Currently outputting single prediction per test input
        y_hat = model(X)
        y_hat = (y_hat > 0.5).float()

        task_error = 1
        if y_hat.shape == y.shape and torch.all(y_hat == y).item():
            task_error = 0

        error += task_error

    return error / len(dataloader)
