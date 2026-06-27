import cv2
import mediapipe as mp

# Setup drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Create a simple green line style to avoid complex dictionary errors
green_line = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

# Turn on webcam
cap = cv2.VideoCapture(0)

# Initialize the Holistic model (Tracks Face, Body, and Hands together)
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("Basic Tracker running. Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image for a selfie-view, then convert to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw Face Mesh
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, green_line, green_line)

        # 2. Draw Body Pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, green_line, green_line)

        # 3. Draw Left Hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, green_line, green_line)

        # 4. Draw Right Hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, green_line, green_line)

        # Show the window
        cv2.imshow('Basic Stardance Tracker', image)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()