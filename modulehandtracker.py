import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, max_hands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initialize the hand detector.
        :param mode: Static mode if True; otherwise, dynamic mode.
        :param max_hands: Maximum number of hands to detect.
        :param detectionCon: Minimum detection confidence threshold.
        :param trackCon: Minimum tracking confidence threshold.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.prevTime = 0  # For FPS calculation

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Detect hands in the image and calculate FPS.
        :param img: Input image (BGR).
        :param draw: Whether to draw detected landmarks.
        :return: Processed image and results object with landmark data.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mphands.HAND_CONNECTIONS)

        # FPS Calculation
        #currTime = time.time()
        #fps = 1 / (currTime - self.prevTime) if (currTime - self.prevTime) > 0 else 0
        #self.prevTime = currTime

        # Display FPS on the image
        #cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #            (0, 255, 0), 2)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Find and return the positions of hand landmarks.
        :param img: Input image.
        :param handNo: Index of the hand to process (default is 0 for the first detected hand).
        :param draw: Whether to draw circles for each landmark.
        :return: A list containing landmark IDs and their x, y pixel coordinates.
        """
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape  # Get image dimensions
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert to pixel values
                lmList.append([id, cx, cy])

                #if draw:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList



def main():
    # Initialize video capture and hand detector
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Find hands and draw landmarks
        img = detector.findHands(img)

        # Get position of landmarks
        lmList = detector.findPosition(img)
        if lmList:
            print(lmList[8])  # Print position of index finger tip

        # Display the video feed
        cv2.imshow("Image", img)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
