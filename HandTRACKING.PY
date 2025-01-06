import cv2
import mediapipe as mp # type: ignore
import time

cap= cv2.VideoCapture(0)

mphands=mp.solutions.hands
hands = mp.Hands()

while True:
    success, imp = cap.read()
    imgRGB=cv2.cvtColor()

    cv2.imshow("image", imp)
    cv2.waitKey(1)