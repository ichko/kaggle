import data
import vis
import models
import logger

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, dataloader, epochs, epoch_end=lambda _: None):
    for epoch in tqdm(range(epochs)):
        tq = tqdm(dataloader)
        for batch in tq:
            loss, info = model.optim_step(batch)

            tq.set_description(f'LOSS: {loss:.5f}')

        epoch_end(epoch)


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


if __name__ == '__main__':
    print('DEVICE:', DEVICE)

    train_dl = data.load_data(
        '.data/training',
        bs=20,
        shuffle=False,
        device=DEVICE,
    )

    val_dl = data.load_data(
        '.data/evaluation',
        bs=5,
        shuffle=False,
        device=DEVICE,
    )

    it = iter(train_dl)
    X, y = next(it)

    saved_model_path = '.models/soft_addressable_cnn.weights'

    model = models.SoftAddressableComputationCNN(input_channels=11)
    model = model.to(DEVICE)
    model.make_persisted(saved_model_path)
    # model = torch.load(f'{saved_model_path}_whole.h5')

    model.summary()

    output = model(X)
    print(output.shape)

    def on_epoch_end(epoch):
        if epoch % 10 == 0:
            train_score = evaluate(model, train_dl)
            val_score = evaluate(model, val_dl)

            print(f'====== EPOCH {epoch} END ======')
            print('FINAL TRAIN SCORE:', train_score)
            print('FINAL VAL SCORE:', val_score)

            model.save()

    model.configure_optim(lr=0.0001)
    train(
        epochs=10_000,
        model=model,
        dataloader=train_dl,
        epoch_end=on_epoch_end,
    )
