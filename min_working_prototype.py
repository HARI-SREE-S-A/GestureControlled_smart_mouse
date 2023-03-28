import cv2
import numpy as np
import handmodule as htm
import time
import autopy
############################# declaring variables
wcam = 640
hcam = 480
ptime = 0
wscr,hscr = autopy.screen.size()
framer = 100 #frame reduction

print(wscr,hscr)

###################### ## finding hand landmark -1

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
detector = htm.handDetector(maxHands=1)






while True:



    success,img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findposition(img)

    if len(lmlist) != 0:
        cv2.rectangle(img, (framer, framer), (wcam - framer, hcam - framer), (255, 0, 255), 2)
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        #print(x1,y1,x2,y2)



    ## get the tip of index and middle finger ## check which fingers are raised
        fingers = detector.fingersup()
        #print(fingers)

    ## only index finger then its in moving mode


        if fingers[1] == 1 and fingers [2] == 0:


        ## converting the co-ordinates

            x3 = np.interp(x1,(framer,wcam-framer),(0,wscr))
            y3 = np.interp(y1,(framer,hcam-framer),(0,hscr))



        ## smoothen the values



        ## move the mouse
            autopy.mouse.move(wscr-x3,y3)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)




    ## check if we are in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length = detector.finddist(img, 12, 8)



    ## distance between the fingers ##click the mouse if the distance is less than threshfold
            if length*100 < 3:
                cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()








    ## frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)


    ## display

    cv2.imshow("harvis",img)
    cv2.waitKey(1)

