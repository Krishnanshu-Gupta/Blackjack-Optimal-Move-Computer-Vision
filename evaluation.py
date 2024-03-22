import glob
from pathlib import Path
import numpy as np
import cv2 as cv
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from extract_corners import extract_corner
from reproject import reproject_playing_card
from recognition import train, extract_hog_features, predict, build_dataset

INPUT_PATH = Path("data/test")
OUTPUT_PATH = Path("data/test")

knn, labels = train()

test_images, test_labels, _ = build_dataset(INPUT_PATH, labels)

ret, results, neighbours, dist = knn.findNearest(np.array(test_images), 3)
test_predict = results[:, 0]

cm = confusion_matrix(test_labels, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

print(classification_report(test_labels, test_predict, target_names=labels))

disp.plot()
plt.show()
