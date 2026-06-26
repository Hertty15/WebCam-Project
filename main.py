import cv2
import mediapipe as mp

#ai setup
#telling mdeiapip we r tracking human poses
mp_pose=mp.solutions.pose
pose=mp_pose.pose(static_image_mode=False, min_detection_confidence=0.5)

#this will draw those weird lines with dots and all u mustve seen on the screen
mp_drawing=mp.solutions.drawing_utils

#turning on cam
#0 means the default cam
cap=cv2.VideoCapture(0)

cap