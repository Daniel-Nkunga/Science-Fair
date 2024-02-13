import dlib
import cv2
import math

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over each detected face
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        
        # Calculate the distance between landmarks 63 and 67
        x63, y63 = landmarks.part(63).x, landmarks.part(63).y
        x67, y67 = landmarks.part(67).x, landmarks.part(67).y
        distanceCenter = math.sqrt((x67 - x63)**2 + (y67 - y63)**2)
        
        x64, y64 = landmarks.part(64).x, landmarks.part(64).y
        x66, y66 = landmarks.part(66).x, landmarks.part(66).y
        distanceRight = math.sqrt((x66 - x64)**2 + (y66 - y64)**2)

        x62, y62 = landmarks.part(62).x, landmarks.part(62).y
        x68, y68 = landmarks.part(68).x, landmarks.part(68).y
        distanceLeft = math.sqrt((x68 - x62)**2 + (y68 - y62)**2)

         # Print the distance (for debugging)
        print("Distance between landmarks 63 and 67:", distanceCenter)
        print("Distance between landmarks 64 and 66:", distanceRight)
        print("Distance between landmakrs 62 and 68:", distanceLeft)
        
        # Draw facial landmarks on the frame
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(frame, f"Center: {distanceCenter:.2f}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Right: {distanceRight:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Left: {distanceLeft:.2f}", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check if distance is less than 20
        if distanceCenter < 20 and distanceRight < 35:
            # Display "Closed" and distances
            cv2.putText(frame, "Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            

    # Resize the frame (optional)
    frame = cv2.resize(frame, (3*800, 3*600))  # Adjust the size as needed

    # Display the resulting frame
    cv2.imshow('Facial Landmarks', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
