import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


folder = "link"
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=2)
offset = 100
imgSize = 500
counter = 0
total_images = 30

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:  
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

            if counter < total_images:
                img_path = f'{folder}/Image_{time.time()}.jpg'
                success = cv2.imwrite(img_path, imgWhite)
                if success:
                    print(f"Image {counter + 1} saved successfully: {img_path}")
                    counter += 1
                else:
                    print(f"Error: Failed to save image at {img_path}")

            if counter >= total_images:
                print(f"{total_images} images saved successfully.")
                break

    cv2.imshow('Image', img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
