from matplotlib import colors
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import numpy as np

import src.utils as utils

# SRC - <https://www.kaggle.com/teddykoker/training-cellular-automata-part-ii-learning-tasks/notebook>
pallet = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA',
    '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'
]

cmap = colors.ListedColormap(pallet)

norm = colors.Normalize(vmin=0, vmax=len(pallet))

# TODO: Show distrib of different tasks kernel picking vectors
# TODO: Plot difference between different kernels
# TODO: Log distance to solutions of train and val set
# TODO: Try with bigger networks


def make_np_image(im):
    r = plt.gcf().canvas.get_renderer()
    x = im.make_image(r)[0]
    return x


def make_task_animation(inputs, outputs, preds_seq, title='', size=2):
    cols, rows = 4, len(inputs)

    fig = plt.figure(figsize=(cols * size, rows * size))

    add_or_get_subplot = utils.memoize(fig.add_subplot)

    def get_actors(preds):
        diffs = abs(outputs - preds)
        for r in range(rows):
            for c, (col, img) in enumerate({
                    'inputs': inputs,
                    'outputs': outputs,
                    'preds': preds,
                    'diffs': diffs,
            }.items()):
                ax = add_or_get_subplot(rows, cols, r * cols + c + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{col}[{r}]')
                yield ax.imshow(img[r], cmap=cmap)

    seq_dim = 1
    seq_range = range(preds_seq.shape[seq_dim])
    imgs = [list(get_actors(preds_seq[:, i])) for i in seq_range]

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(pad=2.5, h_pad=1.2, w_pad=0)

    return animation.ArtistAnimation(fig, imgs, interval=500, blit=True)


def task_to_np_anim(*args, **kwargs):
    ani = make_task_animation(*args, **kwargs)
    ani.new_frame_seq()


def save_task_vid(path, *args, **kwargs):
    ani = make_task_animation(*args, **kwargs)
    ani.save(path)
    plt.close()


def plot_grid(grid):
    fig, ax = plt.subplots(1, 1)

    ax.imshow(grid, cmap=cmap, norm=norm)
    # ax[i].grid(True, which='both', color='lightgrey', linewidth=0.5)

    return fig
