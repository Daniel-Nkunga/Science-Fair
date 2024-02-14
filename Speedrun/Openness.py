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
        
        # Calculate the distance between landmarks 61 and 67
        x61, y61 = landmarks.part(61).x, landmarks.part(61).y
        x67, y67 = landmarks.part(67).x, landmarks.part(67).y
        distanceCenter = math.sqrt((x67 - x61)**2 + (y67 - y61)**2)
        
        x62, y62 = landmarks.part(62).x, landmarks.part(62).y
        x66, y66 = landmarks.part(66).x, landmarks.part(66).y
        distanceRight = math.sqrt((x66 - x62)**2 + (y66 - y62)**2)

        x63, y63 = landmarks.part(63).x, landmarks.part(63).y
        x65, y65 = landmarks.part(65).x, landmarks.part(65).y
        distanceLeft = math.sqrt((x65 - x63)**2 + (y65 - y63)**2)


        #  # Print the distance (for debugging)
        # print("Distance between landmarks 61 and 67:", distanceCenter)
        # print("Distance between landmarks 64 and 66:", distanceRight)
        # print("Distance between landmarks 62 and 66:", distanceLeft)
        
        # Draw facial landmarks on the frame
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n == 61 or n == 67:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            elif n == 62 or n == 66:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)  # Red color for landmarks 61 and 67
            elif n == 63 or n == 65:
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Green color for other landmarks

        cv2.putText(frame, f"Center: {distanceCenter:.2f}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Right: {distanceRight:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Left: {distanceLeft:.2f}", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Check if distance is less than 20
        if distanceCenter < 17 and distanceRight < 17 and distanceLeft < 17:
            # Display "Closed" and distances
            cv2.putText(frame, "Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

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
