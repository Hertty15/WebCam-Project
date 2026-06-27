import cv2
import mediapipe as mp

#ai setup
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

#dedicated body model
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#dedicated face model
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#dedicated hands model
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#cam on
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam!!")
    exit()

print("Digital Twin is running. Press 'q' to quit.")

#main loop
while True:
    ret, frame=cap.read()
    if not ret or frame is None:
        break

    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process each model seperately
    pose_results=pose.process(rgb_frame)
    face_results=face_mesh.process(rgb_frame)
    hand_results=hands.process(rgb_frame)

    try:
        #draw facemesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()

                )

        #draw body
        if pose_results.pose_landmarks:
            for pose_landmarks in pose_results.multi_pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

        #draw hands
        if hand_results.left_hand_landmarks:
            for left_hand_landmarks in hand_results.multi_left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_results.left_hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
        if hand_results.right_hand_landmarks:
            for right_hand_landmarks in hand_results.multi_right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_results.right_hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
    except Exception as e:
        pass #ignore tiny glithces in drawing

    cv2.imshow('Digital Twin', frame)

    if cv2.waitKey(5)&0xFF==ord('q'):
        break

#cleaning
cap.release()
cv2.destroyAllWindows()