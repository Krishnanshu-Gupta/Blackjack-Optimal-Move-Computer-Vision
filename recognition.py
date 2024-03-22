import math
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from basic_strategy import BasicStrategy

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

def calculate_distance(card1, card2):
    x1, y1 = card1
    x2, y2 = card2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

if __name__ == "__main__":
    import sys
    cards_img = cv.imread(sys.argv[1])
    cards = reproject_playing_card(cards_img)
    lst = []
    knn, labels = train()
    for rect, card in cards:
        try:
            corner = extract_corner(card)
            corner = extract_hog_features(corner)
            prediction = predict(knn, labels, corner)
            mid = (np.mean(rect[:, 0]), np.mean(rect[:, 1]))
            lst.append([mid, prediction])

            thickness = round(0.004 * cards_img.shape[0])
            cv.polylines(cards_img, [rect.astype(np.int32)], True, (0, 255, 0), thickness)
            cv.putText(cards_img, prediction, rect.mean(axis=0).astype(np.int32), cv.FONT_HERSHEY_SIMPLEX, 0.002 * cards_img.shape[0], (0, 255, 0), thickness, cv.LINE_AA)

        except RuntimeError as e:
            print(e)

    distances = []
    back_card = None

    # find dealer's back card (card that is flipped over)
    for card in lst:
        if "BACK" in card[1].upper():
            back_card = card[0]
            lst.remove(card)
            break

    if back_card is None:
        raise ValueError("No card with 'Back' label found.")

    # find the closest card to the back card
    min_distance = float('inf')
    closest_card = None
    for card in lst:
        distance = calculate_distance(back_card, card[0])
        if distance < min_distance:
            min_distance = distance
            closest_card = card
    lst.remove(closest_card)

    # get player and dealer hands
    player_hand = []
    dealer_hand = []
    dealer_hand.append(closest_card[1])
    for card in lst:
        if card[1] in ("J", "Q", "K"):
            player_hand.append("10")
        else: player_hand.append(card[1])

    strategy = BasicStrategy(player_hand, dealer_hand)
    move = strategy.recommend()
    print("Player Hand:", player_hand, "| Dealer Upcard:", dealer_hand)
    print("Recommended Move:", move)

    plt.imshow(cv.cvtColor(cards_img, cv.COLOR_BGR2RGB))
    plt.show()
