import mediapipe as mp
import cv2

class HandDetect:
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionConf, 
            min_tracking_confidence=self.trackConf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPos(self, handLms, img):
        lmlist = []
        if handLms:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
        return lmlist

    def countFingers(self, lmlist):
        if len(lmlist) == 0:
            return 0
        
        # Indexes for fingertips
        tipIds = [4, 8, 12, 16, 20]
        
        # Count the number of fingers extended
        fingers = [0] * 5
        
        # Thumb detection (special case)
        if lmlist[4][1] > lmlist[3][1]:  # Thumb extended
            fingers[0] = 1
        
        # Other fingers
        for i in range(1, 5):
            if lmlist[tipIds[i]][2] < lmlist[tipIds[i] - 2][2]:
                fingers[i] = 1
        
        return sum(fingers)

    def getTotalFingers(self, img):
        totalFingers = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmlist = self.findPos(handLms, img)
                totalFingers += self.countFingers(lmlist)
        return min(totalFingers, 10)  # Cap the number of fingers at 10

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    detector = HandDetect()
    while True:
        succes, img = cap.read()
        if not succes:
            print("Error: Could not read frame.")
            break

        img = detector.findHands(img)
        numFingers = detector.getTotalFingers(img)
        
        # Display the total number of fingers or 10 if both hands are shown
        cv2.putText(img, f'Fingers: {numFingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
