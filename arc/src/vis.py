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


def plot_pictures(pictures):
    # fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures), 32))
    fig, axs = plt.subplots(1, len(pictures))
    if len(pictures) == 1:
        axs = [axs]

    for i, pict in enumerate(pictures):
        axs[i].imshow(pict, cmap=cmap, norm=norm)
        axs[i].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axs[i].set_title('')

    return fig


def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict],
                      ['Input', 'Output', 'Predict'])
