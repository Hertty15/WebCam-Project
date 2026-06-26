import cv2
import mediapipe as mp

#ai setup
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing=mp.solutions.drawing_utils

#cam on
cap=cv2.VideoCapture(0)

#cam on check
if not cap.isOpened():
    print("ERROR: Could not switch on webcam!!")
    exit()

print("Posture Tracker is running. Press 'q' on keyboard to quit.")

#main loop
while True:
    #read a frame from cam
    ret, frame=cap.read()
    #check if frame read was success
    if not ret or frame is None:
        print("ERROR: Could not read frame from webcam!!")
        break

    #convert bgr to rgb
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #ai process img
    results=pose.process(rgb_frame)
    #if ai sees body draw skele
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    #show the feed
    cv2.imshow('Posture Tracker', frame)
    #press 'q' tp quit
    if cv2.waitKey(5)&0xFF==ord('q'):
        break

#cleaning
cap.release()
cv2.destroyAllWindows()
print("Tracker stopped.")