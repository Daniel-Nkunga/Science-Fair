import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize video capture
cap = cv2.VideoCapture(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\no2.mp4')  # Replace 'path_to_video_file' with your video file path

special = [13, 14, 82, 87, 312, 317]
closed = 16

mouth_open = False
last_open_frame = -1

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    frame_number = 0
    
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
                # Calculate distances between landmarks
                landmark_13 = (face_landmarks.landmark[13].x * frame.shape[1], face_landmarks.landmark[13].y * frame.shape[0])
                landmark_14 = (face_landmarks.landmark[14].x * frame.shape[1], face_landmarks.landmark[14].y * frame.shape[0])
                distanceCenter = math.sqrt((landmark_13[0] - landmark_14[0])**2 + (landmark_13[1] - landmark_14[1])**2)
                landmark_82 = (face_landmarks.landmark[82].x * frame.shape[1], face_landmarks.landmark[82].y * frame.shape[0])
                landmark_87 = (face_landmarks.landmark[87].x * frame.shape[1], face_landmarks.landmark[87].y * frame.shape[0])
                distanceLeft = math.sqrt((landmark_82[0] - landmark_87[0])**2 + (landmark_82[1] - landmark_87[1])**2)
                landmark_312 = (face_landmarks.landmark[312].x * frame.shape[1], face_landmarks.landmark[312].y * frame.shape[0])
                landmark_317 = (face_landmarks.landmark[317].x * frame.shape[1], face_landmarks.landmark[317].y * frame.shape[0])
                distanceRight = math.sqrt((landmark_312[0] - landmark_317[0])**2 + (landmark_312[1] - landmark_317[1])**2)
                
                # Check if distance is less than 10 and turn landmarks red
                if distanceCenter < closed and distanceLeft < closed and distanceRight < closed:
                    if mouth_open:
                        mouth_open = False
                        last_open_frame = frame_number
                else:
                    if not mouth_open:
                        mouth_open = True
        
        cv2.imshow('Face Dots', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_number += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Last frame where the mouth is open:", last_open_frame)
