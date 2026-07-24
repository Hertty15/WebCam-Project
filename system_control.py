import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- SETUP VOLUME CONTROL ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# --- SETUP MEDIAPIPE ---
mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=0, max_num_hands=1, 
                      min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Stardance Volume Control Active.")
print("Pinch to control volume. Open hand to stop.")
print("Press 'q' to quit.")

# --- SMOOTHING VARIABLES ---
volume_history = []
smooth_factor = 0.2  # Higher = smoother but slower response
current_volume = volume.GetMasterVolumeLevelScalar() * 100  # Start with actual volume
last_volume = current_volume

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw volume track
    cv2.rectangle(img, (50, 100), (85, 400), (100, 100, 100), 3)
    
    # Draw current volume level
    current_vol_height = 400 - int((current_volume / 100) * 300)
    cv2.rectangle(img, (50, current_vol_height), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(current_volume)}%', (40, 430), 
               cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if len(lmList) != 0:
                # Thumb (4) and Index (8)
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                
                # Calculate pinch distance
                length = math.hypot(x2 - x1, y2 - y1)
                
                # --- VOLUME CONTROL ---
                if length < 40:  # Pinched
                    # Map hand Y position to volume range
                    # (100 = top of screen = max volume, 400 = bottom = min volume)
                    vol = np.interp(y2, [100, 400], [maxVol, minVol])
                    
                    # Convert to percentage for display
                    volPer = np.interp(y2, [100, 400], [100, 0])
                    
                    # Add to history for smoothing
                    volume_history.append(volPer)
                    if len(volume_history) > 5:
                        volume_history.pop(0)
                    
                    # Calculate smoothed volume
                    smooth_vol = sum(volume_history) / len(volume_history)
                    
                    # Apply smoothing (gradual change)
                    target_vol = smooth_vol
                    current_volume += (target_vol - current_volume) * smooth_factor
                    
                    # Clamp to 0-100
                    current_volume = max(0, min(100, current_volume))
                    
                    # Set actual volume
                    volume.SetMasterVolumeLevelScalar(current_volume / 100.0, None)
                    
                    # Draw visual feedback
                    cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 15, (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, 'GRABBED', (200, 50), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                else:
                    # When not pinched, keep current volume
                    cv2.putText(img, 'HOVERING', (200, 50), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)
                
                # Draw finger positions
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Stardance Volume Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()