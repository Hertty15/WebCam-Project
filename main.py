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

peace_timer=0

fist_timer=0

is_unlocked=False

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
            #get y coord for index and middle finger
            index_tip=hand_landmarks.landmark[8].y
            index_knuckle=hand_landmarks.landmark[6].y
            middle_tip=hand_landmarks.landmark[12].y
            middle_knuckle=hand_landmarks.landmark[10].y

            #ring finger check
            ring_tip=hand_landmarks.landmark[16].y
            ring_knuckle=hand_landmarks.landmark[14].y
            #pinky finger check
            pinky_tip=hand_landmarks.landmark[20].y
            pinky_knuckle=hand_landmarks.landmark[18].y
            
            #check
            index_up=index_tip<index_knuckle
            middle_up=middle_tip<middle_knuckle
            ring_down=ring_tip>ring_knuckle
            pinky_down=pinky_tip>pinky_knuckle

            #hold timer safety
            if index_up and middle_up and ring_down and pinky_down:
                peace_timer+=1
            else:
                peace_timer=0

            #unlock only if help for 15 frames
            if peace_timer>15:
                print("PEACE SIGN DETECTED - UNLOCKED")
                is_unlocked=True
                peace_timer=0

            #check if all fingers are curled in
            #tip y>knuckle y means tip is lower than knuckle, so finger is curled in
            index_curled=index_tip>index_knuckle
            middle_curled=middle_tip>middle_knuckle
            ring_curled=ring_tip>ring_knuckle
            pinky_curled=pinky_tip>pinky_knuckle

            if index_curled and middle_curled and ring_curled and pinky_curled:
                fist_timer+=1
            else:
                fist_timer=0

            #lock if held for 15 frames
            if fist_timer>15:
                print("FIST DETECTED - LOCKED")
                is_unlocked=False
                fist_timer=0

            if is_unlocked:
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
                #calc rawa vol
                raw_volume=((1-normalized_vol)*100)
                #round to nearest 5
                volume=round(raw_volume/5)*5
                print(f"Volume: {volume}%")
                volume_control.SetMasterVolumeLevelScalar(volume/100.0, None)
            

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()