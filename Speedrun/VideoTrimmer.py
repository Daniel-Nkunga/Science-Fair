import cv2
import mediapipe as mp
import math
from moviepy.editor import VideoFileClip

def get_first_open_frame(video_path):
    # Initialize MediaPipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    closed = 16
    mouth_open = False
    first_open_frame = -1

    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        frame_number = 0

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
                    # Calculate distances between landmarks
                    landmark_13 = (face_landmarks.landmark[13].x * frame.shape[1], face_landmarks.landmark[13].y * frame.shape[0])
                    landmark_14 = (face_landmarks.landmark[14].x * frame.shape[1], face_landmarks.landmark[14].y * frame.shape[0])
                    distanceCenter = math.sqrt((landmark_13[0] - landmark_14[0])**2 + (landmark_13[1] - landmark_14[1])**2)
                    landmark_82 = (face_landmarks.landmark[82].x * frame.shape[1], face_landmarks.landmark[82].y * frame.shape[0])
                    landmark_87 = (face_landmarks.landmark[87].x * frame.shape[1], face_landmarks.landmark[87].y * frame.shape[0])
                    distanceLeft = math.sqrt((landmark_82[0] - landmark_87[0])**2 + (landmark_82[1] - landmark_87[1])**2)
                    landmark_312 = (face_landmarks.landmark[312].x * frame.shape[1], face_landmarks.landmark[312].y * frame.shape[0])
                    landmark_317 = (face_landmarks.landmark[317].x * frame.shape[1], face_landmarks.landmark[317].y * frame.shape[0])
                    distanceRight = math.sqrt((landmark_312[0] - landmark_317[0])**2 + (landmark_312[1] - landmark_317[1])**2)

                    # Check if distance is greater than 10 and turn landmarks red
                    if distanceCenter > closed or distanceLeft > closed or distanceRight > closed:
                        if not mouth_open:
                            mouth_open = True
                            first_open_frame = frame_number
            frame_number += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return first_open_frame

def get_last_open_frame(video_path):
    # Initialize MediaPipe Face Detection
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    special = [13, 14, 82, 87, 312, 317]
    closed = 16

    mouth_open = False
    last_open_frame = -1

    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        frame_number = 0
        
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
                    # Calculate distances between landmarks
                    landmark_13 = (face_landmarks.landmark[13].x * frame.shape[1], face_landmarks.landmark[13].y * frame.shape[0])
                    landmark_14 = (face_landmarks.landmark[14].x * frame.shape[1], face_landmarks.landmark[14].y * frame.shape[0])
                    distanceCenter = math.sqrt((landmark_13[0] - landmark_14[0])**2 + (landmark_13[1] - landmark_14[1])**2)
                    landmark_82 = (face_landmarks.landmark[82].x * frame.shape[1], face_landmarks.landmark[82].y * frame.shape[0])
                    landmark_87 = (face_landmarks.landmark[87].x * frame.shape[1], face_landmarks.landmark[87].y * frame.shape[0])
                    distanceLeft = math.sqrt((landmark_82[0] - landmark_87[0])**2 + (landmark_82[1] - landmark_87[1])**2)
                    landmark_312 = (face_landmarks.landmark[312].x * frame.shape[1], face_landmarks.landmark[312].y * frame.shape[0])
                    landmark_317 = (face_landmarks.landmark[317].x * frame.shape[1], face_landmarks.landmark[317].y * frame.shape[0])
                    distanceRight = math.sqrt((landmark_312[0] - landmark_317[0])**2 + (landmark_312[1] - landmark_317[1])**2)
                    
                    # Check if distance is less than 10 and turn landmarks red
                    if distanceCenter < closed and distanceLeft < closed and distanceRight < closed:
                        if mouth_open:
                            mouth_open = False
                            last_open_frame = frame_number
                    else:
                        if not mouth_open:
                            mouth_open = True
            
            cv2.imshow('Face Dots', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return last_open_frame

def trim_video(start_frame, end_frame, input_video_path, output_video_path):
    # Load the video clip
    video_clip = VideoFileClip(input_video_path)
    
    # Get the duration of each frame in seconds
    frame_duration = 1 / video_clip.fps
    
    # Calculate the start and end time based on frame numbers
    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration
    
    # Trim the video clip
    trimmed_clip = video_clip.subclip(start_time, end_time)
    
    # Write the trimmed video to the output file path
    trimmed_clip.write_videofile(output_video_path, codec="libx264")
    
    # Close the clips
    video_clip.close()
    trimmed_clip.close()

# Example usage:
input_path = "input_video.mp4"
output_path = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\TimmedVids')
start_frame = first_open_frame = get_first_open_frame(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\yes4.mp4')
end_frame = last_open_frame = get_last_open_frame(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\yes4.mp4')
trim_video(start_frame, end_frame, input_path, output_path)

# Example usage:
first_open_frame = get_first_open_frame(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\yes4.mp4')
print("First frame where the mouth is open:", first_open_frame)

last_open_frame = get_last_open_frame(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\yes4.mp4')
print("Last frame where the mouth is open:", last_open_frame)
