import cv2
import mediapipe as mp
import time
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import POINTER, cast

# Initialize pycaw for audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initialize hand detection using MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Variables for FPS calculation
pTime = 0

# Initialize volume control variables
paused_volume = None
is_paused = False

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) != 0:
        # Thumb and index finger
        x1, y1 = lmList[4]
        x2, y2 = lmList[8]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw line and circles
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        if not is_paused:  # Dynamic volume adjustment when not paused
            vol = np.interp(length, [20, 175], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

        # Middle finger tip for pause functionality
        x_mid, y_mid = lmList[12]
        wrist_y = lmList[0][1]

        if y_mid > wrist_y - 50:  # Middle finger close to wrist
            if not is_paused:  # Pause volume
                paused_volume = volume.GetMasterVolumeLevel()
                is_paused = True
        else:
            if is_paused:  # Unpause volume
                is_paused = False

        # Draw rectangle indicator for volume pause status
        status_text = "Volume Paused" if is_paused else "Volume Active"
        rect_color = (0, 0, 255) if is_paused else (0, 255, 0)

        # Draw rectangle with status text
        cv2.rectangle(img, (10, 10), (250, 70), rect_color, -1)
        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Optional feedback for button press effect
        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Volume bar parameters
        volBar = np.interp(length, [20, 175], [400, 150])  # For rectangle height
        volPercent = np.interp(length, [20, 175], [0, 100])  # For percentage display

        # Draw the volume level rectangle
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # Volume bar border
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # Volume level
        cv2.putText(img, f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("Hand Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
