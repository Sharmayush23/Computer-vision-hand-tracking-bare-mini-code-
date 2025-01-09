import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Define the dragRect class
class dragRect():
    def __init__(self, positionCenter, size=(200, 200)):
        self.positionCenter = positionCenter  # center position
        self.size = size  # Size of the rectangle

    def update(self, cursor):
        cx, cy = self.positionCenter
        w, h = self.size
        # Check if the cursor (finger tip) is within the bounds of the rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.positionCenter = cursor  # Update rectangle position to cursor

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8)

# List of rectangles to be created
rectList = []
for x in range(5):  # Creates 5 rectangles
    rectList.append(dragRect([x * 250 + 150, 150]))  # Set initial positions

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands and len(hands) > 0:
        hand = hands[0]  # Get the first detected hand
        lmList, bbox = detector.findPosition(img)  # Get landmarks and bounding box

        # Get the index finger tip's position
        cursor = lmList[8]  # Index finger tip is at index 8
        # Update all rectangles based on cursor (finger tip) position
        for rect in rectList:
            rect.update(cursor)

    # Drawing all the rectangles
    for rect in rectList:
        cx, cy = rect.positionCenter
        w, h = rect.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 0, 255), cv2.FILLED)

    # Show the image with rectangles drawn
    cv2.imshow("Image", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()
