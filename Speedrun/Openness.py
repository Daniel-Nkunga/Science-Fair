import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam
cap = cv2.VideoCapture(0)

special = [13, 14, 82, 87, 312, 317]
closed = 10

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
                # Draw dots on facial landmarks with custom colors
                for i, landmark in enumerate(face_landmarks.landmark):
                    if i in special:  # Landmark 13
                        color = (0, 255, 0)  # Green
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, color, 1)
                        # Print text with spacing at top left corner
                        # cv2.putText(frame, f"Special landmark {i}:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        # cv2.putText(frame, f"({x}, {y})", (10, 50 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    else:
                        color = (255, 0, 0)  # Default color: Red
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, color, 1)

                # Calculate distance between landmarks 13 and 14
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
                    cv2.circle(frame, (int(landmark_13[0]), int(landmark_13[1])), 1, (0, 0, 255), -1)
                    cv2.circle(frame, (int(landmark_14[0]), int(landmark_14[1])), 1, (0, 0, 255), -1)
                    cv2.circle(frame, (int(landmark_82[0]), int(landmark_82[1])), 1, (0, 0, 255), -1)
                    cv2.circle(frame, (int(landmark_87[0]), int(landmark_87[1])), 1, (0, 0, 255), -1)
                    cv2.circle(frame, (int(landmark_312[0]), int(landmark_312[1])), 1, (0, 0, 255), -1)
                    cv2.circle(frame, (int(landmark_317[0]), int(landmark_317[1])), 1, (0, 0, 255), -1)
        
        cv2.imshow('Face Dots', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
