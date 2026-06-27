import cv2
import mediapipe as mp

#helper for the anatomical models
def draw_anatomical_bone(frame, landmarks, start_idx, end_idx, color, thickness=6):
    """Draws a thick bone between two joint landmarks."""
    if landmarks:
        h, w, c = frame.shape
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, pt1, thickness, (255, 255, 255), -1)
        cv2.circle(frame, pt2, thickness, (255, 255, 255), -1)

#setup models
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#face model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#body model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

#hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#cam on
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

print("Stardance Anatomical Tracker running. Press 'q' to quit.")

#main loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #process all models
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    #draw face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    #draw the linse following bones
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        #arms
        draw_anatomical_bone(frame, landmarks, 11, 13, (255, 150, 0), 6)  # Left Shoulder to Elbow
        draw_anatomical_bone(frame, landmarks, 13, 15, (255, 150, 0), 4)  # Left Elbow to Wrist
        draw_anatomical_bone(frame, landmarks, 12, 14, (255, 150, 0), 6)  # Right Shoulder to Elbow
        draw_anatomical_bone(frame, landmarks, 14, 16, (255, 150, 0), 4)  # Right Elbow to Wrist

        #legs
        draw_anatomical_bone(frame, landmarks, 23, 25, (0, 100, 255), 8)  # Left Hip to Knee
        draw_anatomical_bone(frame, landmarks, 25, 27, (0, 100, 255), 5)  # Left Knee to Ankle
        draw_anatomical_bone(frame, landmarks, 24, 26, (0, 100, 255), 8)  # Right Hip to Knee
        draw_anatomical_bone(frame, landmarks, 26, 28, (0, 100, 255), 5)  # Right Knee to Ankle

        #torso
        draw_anatomical_bone(frame, landmarks, 11, 23, (0, 255, 255), 5)  # Left Shoulder to Hip
        draw_anatomical_bone(frame, landmarks, 12, 24, (0, 255, 255), 5)  # Right Shoulder to Hip
        draw_anatomical_bone(frame, landmarks, 11, 12, (0, 255, 255), 5)  # Shoulders
        draw_anatomical_bone(frame, landmarks, 23, 24, (0, 255, 255), 5)  # Hips

    #draw hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    #show result
    cv2.imshow('Stardance Anatomical Tracker', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

#cleaning
cap.release()
cv2.destroyAllWindows()