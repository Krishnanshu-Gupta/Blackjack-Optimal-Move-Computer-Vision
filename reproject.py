import cv2 as cv
import numpy as np

# https://github.com/PyImageSearch/imutils/blob/9f740a53bcc2ed7eba2558afed8b4c17fd8a1d4c/imutils/perspective.py#L9
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    
    D = np.linalg.norm(tl - rightMost, axis=1)
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def get_width_height(pts):
    (tl, tr, br, bl) = pts

    # Finding the maximum width
    maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
 
    # Finding the maximum height.
    maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    return maxWidth, maxHeight

def create_dest_points(width, height):
    # Final destination co-ordinates
    return  np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=int)

def reproject_playing_card(img):
    # Perform closing operatiosn to remove high freqency features (e.g. text)
    kernel = np.ones((5, 5))
    # closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)

    # GrabCut
    '''
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    '''

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5),0)
    _, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Perform additional closing operations on the thresholded image to remove noise
    kernel = np.ones((5, 5))
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel);
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=15)

    # Perform canny edge detection (could help remove the background, probably unnecessary)
    #canny = cv.Canny(morph, 0, 200)

    # Find countours in the image
    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Sort the contours by the largest area (hoping the card is one of the larger features)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]

    cv.drawContours(img, contours, 0, (0,255,0), 3)
    cv.imshow("a", img)
    while cv.waitKey(0) != 27:
        pass

    corners = None
    # Loop through the contours and try to turn them into boxes
    for c in contours:
        # Approximate contours to remove extra points
        epsilon = 0.02 * cv.arcLength(c, True)
        contour = cv.approxPolyDP(c, epsilon, True)
        # Probably a playing card if it has 4 corners
        if len(contour) == 4:
            corners = contour
            break
    if corners is None:
        raise ValueError("Given image doesn't have any rectangular contours!")

    corners = order_points(corners.reshape((4, 2)))
    width, height = get_width_height(corners)
    # If card is landscape flip the width and height and rotate the dest points
    if width > height:
        width, height = height, width
        destination_corners = create_dest_points(width, height)
        destination_corners = np.roll(destination_corners, 1, axis=0)
    else:
        destination_corners = create_dest_points(width, height)

    # Getting the homography
    M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography
    return cv.warpPerspective(img, M, (width, height), flags=cv.INTER_LINEAR)


if __name__ == "__main__":
    import sys
    img = cv.imread(sys.argv[1])
    reprojected = reproject_playing_card(img)
    resized = cv.resize(reprojected, (600, 840), interpolation=cv.INTER_CUBIC)
    cv.imshow("closed", resized)
    while cv.waitKey(0) != 27:
        pass
    cv.destroyAllWindows()
