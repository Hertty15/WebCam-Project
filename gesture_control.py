import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

#safety
#if mouse goes crazy put the mouse in the top left and it will crash
pyautogui.FAILSAFE=True

#get screen resolution
screen_width, screen_height=pyautogui.size()

#camera res
wCam, hCam=640, 480

#mediapip hands setup
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing=mp.solutions.drawing_utils

#cam on
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4, hCam)

pTime=0
plocX, plocY=0,0
curX, curY=0,0

print("Gesture Control Active.")
print("Move Index Finger to move mouse.")
print("Pinch Thumb + Index to click.")
print("Press 'q' to quit.")

while True:
    success, img=cap.read()
    if not success:
        break

    #flip img so it is like mirror
    img=cv2.flip(img, 1)
    h, w, c=img.shape
    rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList=[]
            for id, lm in enumerate(handLms.landmark):
                cx, cy=int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

            if len(lmList) !=0:
                #mouse movement
                x1, y1=lmList[8][1], lmList[8][2]

                #convert cam coord to screen coord
                x3=np.interp(x1, (0,wCam),(0, screen_width))
                y3=np.interp(y1,(0,hCam),(0, screen_height))

                #smoothen mouse moving
                curX=plocX+(x3-plocX)/5
                curY=plocY+(y3-plocY)/5

                pyautogui.moveTo(curX,curY)
                plocX,plocY=curX,curY

                #clicking
                x2, y2=lmList[4][1], lmList[4][2]

                #calc dist betn thumb and index
                length=math.hypot(x2-x1,y2-y1)

                #dist less than 30 px then = pinch
                if length<30:
                    cv2.circle(img, (x1,y1), 15, (0,255,0), cv2.FILLED)#draw green circle
                    pyautogui.click()
                    time.sleep(0.1)#pause a lil to prevent double click

    #show fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()