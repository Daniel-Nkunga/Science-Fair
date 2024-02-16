import cv2
import mediapipe as mp
import numpy as np

def create_face_map(image_path):
    # Load MediaPipe face mesh model
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks
    result = face_mesh.process(image_rgb)
    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]  # Assuming only one face in the image
        vertices = []
        for landmark in face_landmarks.landmark:
            # Convert landmarks to image coordinates
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            vertices.append((x, y))
        
        return vertices
    else:
        return None

def plot_face_vertices(vertices):
    # Create a blank image
    blank_image = np.zeros((600, 800, 3), np.uint8)

    # Plot vertices on the blank image
    for vertex in vertices:
        cv2.circle(blank_image, vertex, 2, (0, 255, 0), -1)

    # Display the image with plotted vertices
    cv2.imshow('Face Vertices', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "face_image.jpg"
    face_map = create_face_map(image_path)
    if face_map:
        print("Face map created successfully!")
        plot_face_vertices(face_map)
    else:
        print("No face detected in the image.")
