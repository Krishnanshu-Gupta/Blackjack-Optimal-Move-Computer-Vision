from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from reproject import reproject_playing_card
from extract_corners import extract_corner

TRAIN_DIR = Path("train/values")

# Initialize HOG descriptor
winSize = (28, 28)
blockSize = (14, 14)
blockStride = (7, 7)
cellSize = (7, 7)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

def extract_hog_features(image):
    # Compute HOG descriptor
    hist = hog.compute(image)
    return hist.flatten()

def train():
    unique_labels = set()

    images = []
    labels = []
    for label_path in TRAIN_DIR.iterdir():
        _, label = label_path.stem.split('-', maxsplit=1)
        for image_path in label_path.iterdir():
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            image = extract_hog_features(image)
            images.append(image)
            labels.append(label)
            unique_labels.add(label)

    unique_labels = sorted(list(unique_labels))
    labels = [unique_labels.index(label) for label in labels]

    knn = cv.ml.KNearest_create()
    knn.train(np.array(images), cv.ml.ROW_SAMPLE, np.array(labels))
    return knn, unique_labels

def predict(knn, labels, image):
    ret, results, neighbours, dist = knn.findNearest(np.array([image]), 3)

    return labels[int(results[0][0])]

if __name__ == "__main__":
    import sys
    cards_img = cv.imread(sys.argv[1])
    cards = reproject_playing_card(cards_img)

    knn, labels = train()
    for rect, card in cards:
        try:
            corner = extract_corner(card)
            corner = extract_hog_features(corner)
            prediction = predict(knn, labels, corner)
            thickness = round(0.004 * cards_img.shape[0])
            cv.polylines(cards_img, [rect.astype(np.int32)], True, (0, 255, 0), thickness)
            cv.putText(cards_img, prediction, rect.mean(axis=0).astype(np.int32), cv.FONT_HERSHEY_SIMPLEX, 0.002 * cards_img.shape[0], (0, 255, 0), thickness, cv.LINE_AA)
        except RuntimeError as e: 
            print(e)

    plt.imshow(cv.cvtColor(cards_img, cv.COLOR_BGR2RGB))
    plt.show()
