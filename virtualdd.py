import cv2
from cvzone.HandTrackingModule import HandDetector

# Define the dragRect class
class dragRect():
    def __init__(self, positionCenter, size=(200, 200)):
        self.positionCenter = positionCenter  # Center position
        self.size = size  # Size of the rectangle

    def update(self, cursor):
        cx, cy = self.positionCenter
        w, h = self.size
        # Check if the cursor (finger tip) is within the bounds of the rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.positionCenter = cursor  # Update rectangle position to cursor


# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# List of draggable rectangles
rectList = []
for x in range(5):  # Create 5 rectangles
    rectList.append(dragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands and landmarks
    hands = detector.findHands(img, draw=True)  # Detect hands and draw them
    if hands:
        hand = hands[0]  # Get the first detected hand
        lmList = hand["lmList"]  # Get list of landmarks
        if lmList:  # Check if landmarks exist
            cursor = lmList[8]  # Tip of the index finger

            # Update rectangles with cursor position
            for rect in rectList:
                rect.update(cursor)

    # Draw the rectangles
    for rect in rectList:
        cx, cy = rect.positionCenter
        w, h = rect.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 0, 255), cv2.FILLED)

    # Show the image
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
