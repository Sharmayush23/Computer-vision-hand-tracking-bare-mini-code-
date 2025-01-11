import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone


class Rectangle:
    def __init__(self, pos_center, size=(200, 200)):
        self.pos_center = pos_center
        self.size = size

    def update(self, cursor):
        cx, cy = self.pos_center
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.pos_center = cursor


# Webcam and detector setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
detector = HandDetector(detectionCon=0.8, maxHands=1)
rect_list = [Rectangle([x * 250 + 150, 150], (200, 200)) for x in range(4)]

alpha = 0.5

while True:
    success, img = cap.read()
    img_new = np.zeros_like(img, np.uint8)
    cursor = None

    # Detect hands
    hands = detector.findHands(img, draw=False)  # No drawing; we handle it
    if hands:
        try:
            # Debugging: Check hands type and content
            print(f"Type of hands: {type(hands)}")
            print(f"Content of hands: {hands}")

            # If hands is a list, access elements using indices
            if isinstance(hands, list) and len(hands) > 0:
                hand = hands[0]
                # Ensure hand is a dictionary and contains "lmList"
                if isinstance(hand, dict) and "lmList" in hand:
                    lm_list = hand["lmList"]
                    if lm_list and len(lm_list) > 8:
                        cursor = lm_list[8]  # Index finger tip landmark
                else:
                    print("Hand data structure is unexpected. Check the debug output above.")
        except Exception as e:
            print(f"Error accessing hand data: {e}")

    # Update rectangles
    for rect in rect_list:
        if cursor:
            rect.update(cursor)
        cx, cy = rect.pos_center
        w, h = rect.size
        cvzone.cornerRect(img_new, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Display with transparency
    blended = cv2.addWeighted(img, 1 - alpha, img_new, alpha, 0)
    cv2.imshow("Drag-and-Drop with Transparency", blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
