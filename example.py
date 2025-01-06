
import cv2
import time
import modulehandtracker as htm

detector = htm.handDetector(max_hands=2, detectionCon=0.7, trackCon=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, handNo=0, draw=False)
    
    if lmList:
        print(f"Index finger tip position: {lmList[8]}")

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
