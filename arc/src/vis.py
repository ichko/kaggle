from matplotlib import colors
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import numpy as np

# SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>
pallet = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA',
    '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'
]

cmap = colors.ListedColormap(pallet)

norm = colors.Normalize(vmin=0, vmax=len(pallet))

# TODO: Show distrib of different tasks kernel picking vectors
# TODO: Plot whole task (single image)
# TODO: Plot difference between different kernels


def normalize_board(board):
    return board.detach().cpu().numpy()


def plot_task_inference(batch, idx, test_preds=None, size=2):
    task = {k: v[idx] for k, v in batch.items()}
    if test_preds is not None:
        task['test_preds'] = test_preds[idx]
        task['test_diff'] = abs(task['test_outputs'] - test_preds[idx])

    max_len = max(task['train_len'], task['test_len'])
    should_show_preds = 'test_preds' in task

    rows = max_len
    cols = 5 + should_show_preds
    col_names = [
        'train_inputs',
        'train_outputs',
        'test_inputs',
        'test_outputs',
    ] + ['test_preds', 'test_diff'] if should_show_preds else []

    fig = plt.figure(figsize=(cols * size, rows * size))

    for r in range(rows):
        for c, col in enumerate(col_names):
            if len(task[col]) <= r: continue
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{col}[{r}]')
            ax.imshow(normalize_board(task[col][r]), cmap=cmap)

    fig.tight_layout()

    return fig


def plot_pictures(pictures):
    fig, axs = plt.subplots(1, len(pictures))
    if len(pictures) == 1:
        axs = [axs]

    for i, pict in enumerate(pictures):
        axs[i].imshow(pict, cmap=cmap, norm=norm)
        axs[i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axs[i].set_title('')

    return fig
