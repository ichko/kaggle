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
# TODO: Plot difference between different kernels
# TODO: Visualize solutions with a gif
# TODO: Log val set solutions
# TODO: Log distance to solutions of train and val set

# TODO: Write readme
# TODO: Try with bigger networks


def normalize_board(board):
    return board.detach().cpu().numpy()


def plot_task_inference(inputs, outputs, preds, size=2):
    diffs = abs(outputs - preds)
    cols, rows = 4, len(inputs)
    fig = plt.figure(figsize=(cols * size, rows * size))

    for r in range(rows):
        for c, (col, img) in enumerate({
                'inputs': inputs,
                'outputs': outputs,
                'preds': preds,
                'diffs': diffs,
        }.items()):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{col}[{r}]')
            ax.imshow(normalize_board(img[r]), cmap=cmap)

    fig.tight_layout()

    return fig


def plot_task(batch, idx, test_preds=None, size=2):
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


def plot_grid(grid):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(normalize_board(grid), cmap=cmap, norm=norm)
    # ax[i].grid(True, which='both', color='lightgrey', linewidth=0.5)

    return fig
