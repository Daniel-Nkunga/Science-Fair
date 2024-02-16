import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Function to detect and align face
def detect_and_align_face(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe FaceMesh
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        # Get the first face landmark points
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Example: Calculate the center of the eyes
        left_eye_center_x = landmarks[159].x * image.shape[1]  # index 159 corresponds to left eye
        left_eye_center_y = landmarks[159].y * image.shape[0]
        right_eye_center_x = landmarks[386].x * image.shape[1]  # index 386 corresponds to right eye
        right_eye_center_y = landmarks[386].y * image.shape[0]
        eye_center = ((left_eye_center_x + right_eye_center_x) // 2, (left_eye_center_y + right_eye_center_y) // 2)
        
        # Example: Align the face based on the eye center
        
        # Crop and normalize the aligned face
        
        # Apply optional enhancement
        
        # Save or use the image
        
        return aligned_face_image
    
    else:
        print("No face detected in the image.")
        return None

# Load an image
image = cv2.imread("face_image.jpg")

# Detect and align the face
aligned_face_image = detect_and_align_face(image)

if aligned_face_image is not None:
    # Display the result
    cv2.imshow("Canonical Face Image", aligned_face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Display the original image
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
