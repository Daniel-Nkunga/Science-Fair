import cv2
import mediapipe as mp
import numpy as np

def create_flattened_face(image_path):
    # Initialize MediaPipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read the image from the file
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Mesh
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Create a blank mask
        mask = np.zeros_like(image)

        # Draw landmarks on the mask
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(mask, (x, y), 2, (255, 255, 255), -1)

        # Convert the mask to grayscale
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate the contour to get the convex hull
        convex_hull = cv2.convexHull(contours[0])

        # Create a blank canvas
        canvas = np.zeros_like(image)

        # Draw the convex hull on the canvas
        cv2.drawContours(canvas, [convex_hull], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Save the flattened image
        cv2.imwrite("flattened_image.png", canvas)

        # Display the flattened image
        cv2.imshow('Flattened Image', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Old Files\face.jpg')  # Replace 'path_to_your_image.jpg' with the path to your image
    create_flattened_face(image_path)




#(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Old Files\face.jpg')