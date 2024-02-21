import cv2
import mediapipe as mp

# Define the landmark indices to be displayed
landmark_indices = list(range(1, 469))

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect facial landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw dots on selected facial landmarks
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in landmark_indices:
                        landmark_x = int(landmark.x * frame.shape[1])
                        landmark_y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (landmark_x, landmark_y), 1, (255, 255, 255), 2)

                        # Draw a red square centered around landmark 4
                        if idx == 4:
                            square_size = 300
                            top_left_x = int(landmark_x - square_size / 2)
                            top_left_y = int(landmark_y - square_size / 2)
                            bottom_right_x = int(landmark_x + square_size / 2)
                            bottom_right_y = int(landmark_y + square_size / 2)
                            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)

        cv2.imshow('Face Dots', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
