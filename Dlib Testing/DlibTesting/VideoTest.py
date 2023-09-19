import dlib
import cv2
import bz2
import os

# Check if the .dat file exists, if not, extract it from the .bz2 archive
dat_file = "shape_predictor_68_face_landmarks.dat"
bz2_file = "shape_predictor_68_face_landmarks.dat.bz2"

if not os.path.exists(dat_file):
    with open(dat_file, 'wb') as f_out, bz2.BZ2File(bz2_file, 'rb') as f_in:
        for data in iter(lambda: f_in.read(100 * 1024), b''):
            f_out.write(data)

# Initialize the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat_file)


# Initialize the video capture with your video file
video_file = 'DlibVideoTest.mp4'
cap = cv2.VideoCapture(video_file)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through detected faces
    for face in faces:
        # Get the face landmarks
        landmarks = predictor(gray, face)

        # Draw face landmarks on the frame
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the frame with face landmarks
    cv2.imshow("Face Landmarks", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
