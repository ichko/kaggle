import data
import vis
import model

import matplotlib.pyplot as plt

if __name__ == '__main__':
    sample = data.TRAIN[0]['train'][0]

    vis.plot_sample(sample)
