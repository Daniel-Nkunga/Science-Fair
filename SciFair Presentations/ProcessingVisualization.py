import cv2
import mediapipe as mp
import time
from collections import deque

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
    
    prev_time = 0
    show_lips_landmarks = False
    show_eyebrows_landmarks = False
    
    # Define lips and eyebrows landmarks
    lips_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    left_eyebrow_landmarks = [276, 283, 282, 295, 300, 293, 334, 296, 336]
    right_eyebrow_landmarks =  [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

    # Initialize deque to store FPS values for the last 2 seconds
    fps_values = deque(maxlen=60)
    
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
                if show_lips_landmarks and show_eyebrows_landmarks:
                    # Draw lips and eyebrows landmarks
                    for idx in lips_landmarks:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
                    for idx in left_eyebrow_landmarks + right_eyebrow_landmarks:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
                elif show_lips_landmarks:
                    # Draw only lips landmarks
                    for idx in lips_landmarks:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)
                elif show_eyebrows_landmarks:
                    # Draw only eyebrows landmarks
                    for idx in left_eyebrow_landmarks + right_eyebrow_landmarks:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
                else:
                    # Draw all facial landmarks
                    for i, landmark in enumerate(face_landmarks.landmark):
                        color = (255, 255, 255)  # Default color: Red
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, color, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Append current FPS to the deque
        fps_values.append(fps)
        
        # Calculate average FPS over the last 2 seconds
        avg_fps = sum(fps_values) / len(fps_values)
        
        # Display FPS in top right corner
        cv2.putText(frame, f'Avg FPS: {int(avg_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Face Dots', frame)
        
        # Toggle to display only lips landmarks on space bar press
        key = cv2.waitKey(1)
        if key == ord(' '):
            if not show_lips_landmarks and not show_eyebrows_landmarks:
                show_lips_landmarks = True
            elif show_lips_landmarks and not show_eyebrows_landmarks:
                show_lips_landmarks = True
                show_eyebrows_landmarks = True
            else:
                show_lips_landmarks = False
                show_eyebrows_landmarks = False
        # Break loop on 'q' key press
        elif key & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
