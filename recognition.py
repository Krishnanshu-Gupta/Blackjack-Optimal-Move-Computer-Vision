from pathlib import Path

import cv2 as cv
import numpy as np

from reproject import reproject_playing_card
from extract_corners import extract_corner

TRAIN_DIR = Path("train/values")

def train():
    unique_labels = set()

    images = []
    labels = []
    for label_path in TRAIN_DIR.iterdir():
        _, label = label_path.stem.split('-', maxsplit=1)
        for image_path in label_path.iterdir():
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            image = np.float32(image.flatten()) / 255.0
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
    for card in cards:
        try:
            corner = extract_corner(card)
            corner = np.float32(corner.flatten()) / 255.0
            prediction = predict(knn, labels, corner)
            print(prediction)
        except RuntimeError as e: 
            print(e)
