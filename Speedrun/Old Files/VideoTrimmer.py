import cv2
import mediapipe as mp
import math
from moviepy.editor import VideoFileClip
import os

def get_first_open_frame(frames):
    # Initialize MediaPipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh

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

        for frame in frames:
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

    return first_open_frame

def get_last_open_frame(frames):
    # Initialize MediaPipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh

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

        for frame in frames:
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
                    if distanceCenter < closed or distanceLeft < closed or distanceRight < closed:
                        if mouth_open:
                            mouth_open = False
                            last_open_frame = frame_number
                    else:
                        if not mouth_open:
                            mouth_open = True
            frame_number += 1

    return last_open_frame

def trim_video(start_frame, end_frame, frames, output_video_path):
    # Get the duration of each frame in seconds
    frame_duration = 1 / 30  # Assuming 30 fps
    
    # Calculate the start and end time based on frame numbers
    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration
    
    # Trim the frames
    trimmed_frames = frames[start_frame:end_frame + 1]
    
    # Write the trimmed frames to a video file
    height, width, _ = trimmed_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    
    for frame in trimmed_frames:
        out.write(frame)
    
    # Release the video writer
    out.release()

# Input folder path
input_folder = r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\Train\Nice'

# Output folder path
output_folder = r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\TrimmedVids'

# Ensure the output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{filename}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(input_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # Process each video file
        start_frame = get_first_open_frame(frames)
        end_frame = get_last_open_frame(frames)
        
        print(f"Trimming {filename}...")
        
        try:
            trim_video(start_frame, end_frame, frames, output_path)
            print(f"{filename} trimmed. That's a wrap!")
        except Exception as e:
            print(f"Error occurred while trimming {filename}: {e}")