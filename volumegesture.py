import cv2
import mediapipe as mp
import time
import modulehandtracker as mht

detector=mht.handDetector()

wcam,hcam=720,640
cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ptime=0

while True:

    success , img = cap.read()

    hands=detector.findHands(img , draw=True)


    ctime=time.time()
    fps=1/(ctime - ptime)
    ptime=ctime
    fps_text = f'FPS: {int(fps)}'
    text_position = (50, 50)  # Position at 50px from top-left
    text_font = cv2.FONT_HERSHEY_SIMPLEX  # Use a simple font
    text_scale = 1.5  # Font size
    text_color = (0, 255, 0)  # Green text
    text_thickness = 3  # Bold tex

    cv2.putText(img, fps_text, text_position, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    
    
    
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()