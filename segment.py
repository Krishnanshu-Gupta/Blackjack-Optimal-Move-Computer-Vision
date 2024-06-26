import cv2 as cv
import numpy as np

def segment_cards(img, K = 3):
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
     
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_PP_CENTERS 
         
    # Apply KMeans
    w, h, c = img.shape
    img_flat = np.float32(img.reshape(w * h, c))
    _, labels, _ = cv.kmeans(img_flat, K, None, criteria, 10, flags)

    # Convert image to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV).reshape((w * h, 3))
    scores = []
    for i in range(K):
        labels = labels.reshape(-1)
        mask = labels == i
        masked = img_hsv[mask]
        # Caculate "score" for each cluster as (1.0 - average saturation) + average luminance
        avg_sat = masked[:, 1].mean()
        avg_val = masked[:, 2].mean()
        scores.append((255.0 - avg_sat) + avg_val)

    # Convert the shape of the labels to match the image and select the cluster with max score
    labels_mask = labels.reshape(img.shape[:2])
    return img * (labels_mask == np.argmax(scores))[:, :, np.newaxis], labels_mask

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from skimage import color

    img = cv.imread(sys.argv[1])
    cards, labels = segment_cards(img)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(color.label2rgb(labels, img))
    plt.savefig("segment.png")
    plt.show()
