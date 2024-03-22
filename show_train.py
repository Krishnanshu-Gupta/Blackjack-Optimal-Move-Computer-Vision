#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

TRAIN_DIR = Path("train/values")
SAMPLES = 10

labels_paths = list(TRAIN_DIR.iterdir())
paths = np.array([list(dir.iterdir())[:SAMPLES] for dir in labels_paths]).flatten()

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(len(labels_paths), SAMPLES))

for ax, path in zip(grid, paths):
    im = plt.imread(path)
    ax.imshow(im, cmap="grey")

plt.savefig("train.png")
plt.show()
