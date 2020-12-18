import torch
import torch.nn as nn
import numpy as np

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


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((11, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(11):
        img[i] = inp == i
    return img


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
