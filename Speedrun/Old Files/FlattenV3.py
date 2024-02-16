import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Face Detection
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Show the image with landmarks
        cv2.imshow('Face Landmarks Detection', image)

        # Exit the program when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
