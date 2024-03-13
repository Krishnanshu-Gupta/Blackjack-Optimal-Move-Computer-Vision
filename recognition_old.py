from pathlib import Path
import sys
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from reproject import reproject_playing_card

def load_templates(template_dir: Path):
    paths = glob.glob(str(template_dir / '*-template.*'))
    templates = {}
    for path in paths:
        path = Path(path)
        template = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        target = path.stem.split("-", maxsplit=1)[0]
        templates[target] = template
    return templates

def get_ccoeff(source, template):
    res = cv.matchTemplate(source, template, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val

templates = load_templates(Path("templates"))

img = cv.imread(sys.argv[1])
card = reproject_playing_card(img)

resized = cv.resize(card, (600, 840), interpolation=cv.INTER_CUBIC)
gray = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)

ccoeffs = [get_ccoeff(gray, template) for template in templates.values()]

best_target = list(templates)[np.argmax(ccoeffs)]

print(f"Probably: {best_target}")
#plt.imshow(gray, cmap='gray')
#plt.show()
