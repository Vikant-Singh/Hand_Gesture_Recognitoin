from itertools import groupby
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
cap = cv2.VideoCapture(0)   # 0 is id number
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
str =[]
prv = []
labels = ['1','2','3','4','5','A','B','C','D','E']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)
            str += labels[index]

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)    # for centering the image  matrix in white background
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            str += labels[index]
        cv2.putText(imgOutput, labels[index], (x, y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),3)
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('e') :          # delay of 1 millisecond
        break
print("Printing string")
res = [i[0] for i in groupby(str)]
print(res)
cap.release()
cv2.destroyAllWindows()


