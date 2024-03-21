import glob
from pathlib import Path

import cv2 as cv

EXPECTED_CARD_ASPECT = 1 / 1.4
EPSILON = 0.2

def extract_corner(card):
    h, w, _ = card.shape

    if EXPECTED_CARD_ASPECT - w / h > EPSILON:
        raise RuntimeError(f"invalid card aspect {w / h}, probably invalid")

    # get top left corner of each card
    # just hardcoded, could use contour detection to find biggest thing in a corner
    corner = card[int(0.03 * h):int(0.15 * h), int(0.01 * w):int(0.13 * w)]
    ch, cw, _ = corner.shape
    if ch == 0 or cw == 0:
        raise RuntimeError(f"image too small!")

    corner = cv.cvtColor(corner, cv.COLOR_RGB2GRAY)
    corner = cv.inRange(corner, 0, 125)
    corner = cv.resize(corner, (28, 28), interpolation=cv.INTER_CUBIC)
    return corner

if __name__ == "__main__":
    INPUT_PATH = Path("train/cards")
    OUTPUT_PATH = Path("train/corners")

    card_paths = glob.glob(str(INPUT_PATH / "*.png"))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for path in card_paths:
        path = Path(path)
        print(path.stem)
        card = cv.imread(str(path))

        corner = extract_corner(card)
        cv.imwrite(str(OUTPUT_PATH / f"{path.stem}-tl.png"), corner)
