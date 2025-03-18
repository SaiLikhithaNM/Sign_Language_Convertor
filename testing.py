import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

engine=pyttsx3.init()
def speakText(command):
    engine.say(command)
    engine.runAndWait()
while(1):
    model_path = "\\Sign-Language-detection\\keras_model.h5"
    label_path = "\\Sign-Language-detection\\labels.txt"

    print("Loading classifier...")
    classifier = Classifier(model_path, label_path)
    print("Classifier loaded.")


    detector = HandDetector(maxHands=1)

    offset = 100
    imgSize = 300  
    labels = ["Indian", "Capital", "Hello", "Love", "Namasthe", "Name", "Thank you", "What", "You"]


    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Webcam opened successfully.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop is not None:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                try:
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"Predictions: {prediction}")
                    print(f"Predicted class: {labels[index]}")
                    speakText(f"{labels[index]}")
                except Exception as e:
                    print(f"Error during prediction: {e}")

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
            else:
                print("Error: Cropped image is None")

        cv2.imshow('Image', imgOutput)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
