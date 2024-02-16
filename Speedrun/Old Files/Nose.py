import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam
cap = cv2.VideoCapture(0)


nose = [1, 2, 3, 4, 5, 6, 20, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 102, 115, 122, 125, 129, 131, 134, 141, 166, 168, 195, 196, 197, 198, 203, 209, 218, 219, 220, 235, 236, 237, 238, 239, 240, 241, 242, 248, 249, 250, 278, 279, 289, 290, 294, 305, 309, 314, 326, 328, 331, 341, 342, 344, 429, 432, 436, 438, 439, 440, 443, 448, 452, 455, 456, 457, 458, 459, 460, 461, 462]

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
                # Draw dots on facial landmarks related to lips (indices: 61-67)
                for idx in nose:
                    landmark = face_landmarks.landmark[idx]
                    landmark_x = int(landmark.x * frame.shape[1])
                    landmark_y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (landmark_x, landmark_y), 1, (255, 0, 0), -1)
                
                # # Print the number of each landmark
                # for idx, landmark in range(61, 68):
                #     landmark_x = int(landmark.x * frame.shape[1])
                #     landmark_y = int(landmark.y * frame.shape[0])
                #     cv2.putText(frame, str(idx), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Face Dots', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
