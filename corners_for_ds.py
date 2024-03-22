import glob
from pathlib import Path
import cv2 as cv

from extract_corners import extract_corner
from reproject import reproject_playing_card
from recognition import train, extract_hog_features, predict

INPUT_PATH = Path("images")
OUTPUT_PATH = Path("data/test")

LABEL_DIRS = {
    "A": "01-A",
    "2": "02-2",
    "3": "03-3",
    "4": "04-4",
    "5": "05-5",
    "6": "06-6",
    "7": "07-7",
    "8": "08-8",
    "9": "09-9",
    "10": "10-10",
    "J": "11-J",
    "Q": "12-Q",
    "K": "13-K",
    "BACK": "99-BACK",
}

card_paths = glob.glob(str(INPUT_PATH / "easy" / "*.jpg"))
card_paths.extend(glob.glob(str(INPUT_PATH / "medium" / "*.jpg")))

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

knn, labels = train()

for path in card_paths:
    path = Path(path)
    print(path)
    cards_img = cv.imread(str(path))
    cards = reproject_playing_card(cards_img)

    for i, (rect, card) in enumerate(cards):
        try:
            corner = extract_corner(card)
            features = extract_hog_features(corner)
            label = predict(knn, labels, features)
            
            label_dir = OUTPUT_PATH / LABEL_DIRS[label]
            label_dir.mkdir(exist_ok=True)
            cv.imwrite(str(label_dir / f"{path.stem}-{i}-tl.png"), corner)
        except RuntimeError as e:
            print(e)
