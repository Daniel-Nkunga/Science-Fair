import dlib
import cv2

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # You'll need to download this file

# Initialize the webcam (you can change the source if using a different camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks for the detected face
        landmarks = predictor(gray, face)
        
        # Extract the coordinates for the eyes, eyebrows, lips, and nose
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y
        left_eyebrow = landmarks.part(17).x, landmarks.part(17).y
        right_eyebrow = landmarks.part(26).x, landmarks.part(26).y
        nose = landmarks.part(30).x, landmarks.part(30).y
        lips = landmarks.part(48).x, landmarks.part(48).y

        # Draw circles for the facial features
        cv2.circle(frame, left_eye, 2, (0, 0, 255), -1)
        cv2.circle(frame, right_eye, 2, (0, 0, 255), -1)
        cv2.circle(frame, left_eyebrow, 2, (0, 0, 255), -1)
        cv2.circle(frame, right_eyebrow, 2, (0, 0, 255), -1)
        cv2.circle(frame, nose, 2, (0, 0, 255), -1)
        cv2.circle(frame, lips, 2, (0, 0, 255), -1)

        # Draw a basic outline of the face
        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
        cv2.polylines(frame, [landmarks_list], isClosed=True, color=(0, 255, 0), thickness=1)

    # Display the frame
    cv2.imshow("Facial Feature Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
