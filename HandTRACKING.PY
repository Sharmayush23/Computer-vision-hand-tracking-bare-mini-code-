import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands module
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

# Setup a counter for FPS calculation
prevTime = 0
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert BGR to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image with Mediapipe
    results = hands.process(imgRGB)

    # Draw landmarks if any are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h , w, c = img.shape
                cx , cy = int(lm.x * w), int(lm.y * h)
                print(id,cx,cy)
                if id == 0:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    # Calculate FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
    prevTime = currTime

    # Display FPS
    fps_text = f'FPS: {int(fps)}'
    text_position = (50, 50)  # Position at 50px from top-left
    text_font = cv2.FONT_HERSHEY_SIMPLEX  # Use a simple font
    text_scale = 1.5  # Font size
    text_color = (0, 255, 0)  # Green text
    text_thickness = 3  # Bold text

    cv2.putText(img, fps_text, text_position, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Image", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
