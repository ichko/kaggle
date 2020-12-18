import data
import vis
import model

import matplotlib.pyplot as plt

if __name__ == '__main__':
    task = data.TRAIN['db3e9e38']['train']
    sample = task[0]
    # vis.plot_sample(sample)

    dl = data.TRAIN_DL(bs=1, shuffle=False)
    it = iter(dl)
    batch = next(it)
    batch = next(it)

    # model, num_steps, losses = model.solve_task(task)
