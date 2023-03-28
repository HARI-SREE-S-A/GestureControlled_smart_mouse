import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):

        self.tipIds = [4,8,12,16,20]
        self.lmList = []

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findposition(self, img, handno=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            self.results.multi_hand_landmarks[handno]
            myhand = self.results.multi_hand_landmarks[handno]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                self.lmList.append([id, cx, cy])

        return self.lmList

    def fingersup(self):
        self.tipIds = [4,8,12,16,20]
        fingers = []

        #thumb
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]- 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1,5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]- 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return  fingers

        return fingers


def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findposition(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        if len(lmList) != 0:
            print(lmList[4])

        cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        cv2.imshow("new_harvis", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

