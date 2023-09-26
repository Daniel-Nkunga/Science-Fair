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

def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords

# Initialize the camera capture
cap = cv2.VideoCapture(0)

#There needs to be a for loop here, maybe a while loop for when the video is running
frame_buffer = []
frame_buffer_color = []
while(True):
      success, frame = cap.read()
      if not success: 
            break
      gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY) #For somereason this has to add Bayer to it
      frame_buffer.append(gray)
      frame_buffer_color.append(frame)
cap.release()

# Obtain face landmark information
    landmark_buffer = []        # A list to hold face landmark information
    for (i, image) in enumerate(frame_buffer):          # Iterate on frame buffer
        face_rects = face_detector(image,1)             # Detect face
        if len(face_rects) < 1:                 # No face detected
            print("No face detected: ",vid_path)
            logfile.write(vid_path + " : No face detected \r\n")
            break
        if len(face_rects) > 1:                  # Too many face detected
            print("Too many face: ",vid_path)
            logfile.write(vid_path + " : Too many face detected \r\n")
            break
        rect = face_rects[0]                    # Proper number of face
        landmark = landmark_detector(image, rect)   # Detect face landmarks
        landmark = shape_to_list(landmark)
        landmark_buffer.append(landmark)

