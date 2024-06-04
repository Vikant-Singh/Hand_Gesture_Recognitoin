
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
counter = 0

cap = cv2.VideoCapture(0)   # 0 is id number
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "images/E"
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]     # because we are using single hand
        x, y, w, h = hand['bbox']
        # we are going to create an image with white background with the matrix and it is support by numpy
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255      #uint8 means 0 - 255 uses for color * by 255 because pixel value is 1
        # we are defineing the starting height and ending height and also starting and ending width and offset for cropbox good behaviour
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # putting imgcrop matrix on img white means to removing the backgound
        imgCropShape = imgCrop.shape


        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)    # for centering the image  matrix in white background
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)    # for centering the image  matrix in white background
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # delay of 1 millisecond
    if key == ord("c"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # time.time give unique value
        print(counter)

