import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import POINTER, cast

# Initialize Audio Utilities
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Mediapipe Hand Initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Streamlit Setup
st.set_page_config(page_title="Webcam Volume Control", layout="wide")
st.title("Webcam Volume Control with Hand Gestures")
st.sidebar.header("Controls")
st.sidebar.text("Use buttons below to toggle webcam")

if "toggle" not in st.session_state:
    st.session_state["toggle"] = False

# Control buttons
start_button = st.sidebar.button("Start Webcam", key="start")
stop_button = st.sidebar.button("Stop Webcam", key="stop")
if start_button:
    st.session_state["toggle"] = True
if stop_button:
    st.session_state["toggle"] = False

# Streamlit Columns
col1, col2 = st.columns(2)
col1.subheader("Webcam Feed")
col2.subheader("Volume Level Indicator")
FRAME_WINDOW = col1.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Webcam Feed", use_container_width=True)
vol_bar_placeholder = col2.empty()

if st.session_state["toggle"]:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    pTime = 0
    is_paused = False
    while st.session_state["toggle"]:
        success, img = cap.read()
        if not success:
            FRAME_WINDOW.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Webcam Feed", use_container_width=True)
            st.error("Webcam feed not detected!")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        lmList = []
        results = hands.process(imgRGB)
        volPercent = 0  # Default value for cases with no hand detection
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                # Thumb, Index, and Middle Finger Points
                x1, y1 = lmList[4]   # Thumb
                x2, y2 = lmList[8]   # Index
                x_mid, y_mid = lmList[12]  # Middle Finger

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint

                # Draw Circles
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Thumb
                cv2.circle(img, (x2, y2), 15, (0, 255, 255), cv2.FILLED)  # Index
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Midpoint
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (x_mid, y_mid), 7, (0, 255, 0), cv2.FILLED)  # Middle Finger

                # Calculate distance between thumb and index finger
                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [20, 175], [minVol, maxVol])
                volPercent = np.interp(length, [20, 175], [0, 100])
                volBar = np.interp(length, [20, 175], [400, 150])

                if length < 30:
                   cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Volume bar parameters
                volBar = np.interp(length, [20, 175], [400, 150])  # For rectangle height
                volPercent = np.interp(length, [20, 175], [0, 100])  # For percentage display

                # Update the volume if not paused
                if not is_paused:
                    volume.SetMasterVolumeLevel(vol, None)

                # Draw Volume Level Bar
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # Border
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPercent)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Update Streamlit frontend
        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Webcam Feed", use_container_width=True)
        vol_bar_placeholder.metric("Volume Level", f"{int(volPercent)}%")

    cap.release()
    cv2.destroyAllWindows()
else:
    FRAME_WINDOW.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Webcam Feed", use_container_width=True)
