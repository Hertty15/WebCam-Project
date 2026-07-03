import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#setup vol control
devices=AudioUtilities.GetSpeakers()
volume_control=devices.EndpointVolume.QueryInterface(IAudioEndpointVolume)

#intialize mediapipe hands
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()

#cam on
cap=cv2.VideoCapture(0)

while cap.isOpened():
    success, frame=cap.read()
    if not success:
        break

    #flip img so it is like mirror
    img=cv2.flip(frame, 1)
   
   #convert to rgb for mediapipe
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #find hands
    results=hands.process(rgb)

#draw hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_landmarks=results.multi_hand_landmarks[0]
            h, w, c=frame.shape
            #get xand y coord of center of hand
            cx=int(hand_landmarks.landmark[9].x*w)
            cy=int(hand_landmarks.landmark[9].y*h)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            #def usbale area
            min_y=int(h*0.1)
            max_y=int(h*0.9)
            usable_height=max_y-min_y
            #calm cy
            cy_clamped=max(min_y, min(cy, max_y))
            #calc vol
            normalized_vol=(cy_clamped-min_y)/usable_height
            volume=int((1-normalized_vol)*100)
            print(f"Volume: {volume}%")
            volume_control.SetMasterVolumeLevelScalar(volume/100.0, None)
            

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()