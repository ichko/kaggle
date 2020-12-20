import data
import vis
import models

import matplotlib.pyplot as plt


def trivial_model(batch):
    return [batch['test_outputs']] * 3


if __name__ == '__main__':
    task = data.TRAIN['db3e9e38']['train']
    sample = task[0]

    # vis.plot_sample(sample)

    score = models.score(model=trivial_model,
                         dataloader=data.TRAIN_DL(bs=1, shuffle=False))

    print('SCORE', score)  # less is better
    dl = data.TRAIN_DL(bs=1, shuffle=False)
    it = iter(dl)
    batch = next(it)
    batch = next(it)

    model = models.SoftAddressableComputationCNN(input_channels=11)
    model.summary()

    output = model(batch)
    print(output.shape)

    # model, num_steps, losses = model.solve_task(task)
