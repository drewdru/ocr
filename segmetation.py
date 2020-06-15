import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt

def lines_segmentation(image):
    invert = 255 - image
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(invert, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 2
    H, W = invert.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    # show result
    # result = invert.copy()
    # result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    # for y in uppers:
    #     cv2.line(result, (0,y), (W, y), (255,0,0), 1)
    # for y in lowers:
    #     cv2.line(result, (0,y), (W, y), (0,255,0), 1)
    # cv2.imshow("line", result)
    # cv2.waitKey(0)

    lines = zip(uppers, lowers)
    for line in lines:
        yield invert[line[0]:line[1]]

    # # Get histogram
    # invert = 255 - image
    # proj = np.sum(invert, 1)
    # max_value = np.max(proj)

    # # Get peaks
    # # TODO: How to calculate best KERNEL_SIZE?
    # KERNEL_SIZE = 35
    # proj = medfilt(proj, 35)
    # lower_peaks, _ = find_peaks(-proj+max_value, height=proj.shape[0], distance=KERNEL_SIZE)

    # # split by lower peaks
    # for row in np.split(invert, lower_peaks)[1:-1]:
    #     yield row

def word_segmentation(image):
    median = medfilt(image, 5)

    # apply some dilation and erosion to join the gaps
    thresh = cv2.dilate(median, None, iterations = 10)
    thresh = cv2.erode(thresh, None, iterations = 10)

    proj = np.sum(thresh, 0)
        
    is_word = False
    words = []
    for index, value in enumerate(proj):
        if value > 0 and not is_word:
            is_word = True
            words.append(index)
        if is_word and value == 0:
            is_word = False
            words.append(index)

    for row in np.split(image, words, 1)[1:-1:2]:
        yield row

def character_segmentation(image):
    # TODO: apply some dilation and erosion to join the gaps? fixing blur could help
    # TODO: if this doesn't help than try to segmentate by histogram peaks
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    ) 
    coord = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if h<15 or w<15:
            # TODO: get spechial characters (. , - ' ' "")
            continue  
        coord.append((x,y,w,h))
    coord.sort(key=lambda tup:tup[0])

    # cv2.imshow("image", image)
    for cor in coord:
        [x,y,w,h] = cor
        character = image[y:y+h, x:x+w]
        yield character


# def character_segmentation(image):
#     median = medfilt(image, 5)

#     # apply some dilation and erosion to join the gaps
#     thresh = cv2.dilate(median, None, iterations = 5)
#     thresh = cv2.erode(thresh, None, iterations = 5)

#     proj = np.sum(thresh, 0)
#     max_value = np.max(proj)

#     # Get peaks
#     # TODO: How to calculate best KERNEL_SIZE?
#     KERNEL_SIZE = 25
#     proj = medfilt(proj, KERNEL_SIZE)
#     lower_peaks, _ = find_peaks(-proj+max_value, height=proj.shape[0], distance=KERNEL_SIZE)

#     # split by lower peaks
#     cv2.imshow("image", image)
#     cv2.imshow("thresh", thresh)
#     for columns in np.split(image, lower_peaks, 1):
#         # cv2.imshow("columns", columns)
#         # cv2.waitKey(0)
#         yield columns