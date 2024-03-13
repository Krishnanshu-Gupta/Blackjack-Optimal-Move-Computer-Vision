import glob
from pathlib import Path

import cv2 as cv

from reproject import reproject_playing_card

INPUT_PATH = Path("images")
OUTPUT_PATH = Path("train/cards")

if __name__ == "__main__":
    image_paths = glob.glob(str(INPUT_PATH / "*.jpg"))
    for path in image_paths:
        print(path)
        path = Path(path)
        image = cv.imread(str(path))
        cards = reproject_playing_card(image)
        for i, card in enumerate(cards):
            cv.imwrite(str(OUTPUT_PATH / f"{path.stem}-{i}.png"), card)
