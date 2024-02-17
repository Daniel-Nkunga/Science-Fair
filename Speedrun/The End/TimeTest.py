import cv2
import os
import mediapipe as mp
import csv
import multiprocessing
import time

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to process each video
def process_video(video_path, output_path, focus):
    print(f"Processing video: {video_path}")
    # Initialize VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    # CSV file to store the coordinates
    csv_file_path = os.path.splitext(output_path)[0] + 'L.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        header = ['Landmark']
        for i in range(len(focus) * 2):
            header.append(f'Coordinate {i+1}')
        csv_writer.writerow(header)

        # Initialize MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            for landmark_idx in focus:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video to start
                row = [landmark_idx]
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
                            # Record landmark coordinates for the current frame
                            landmark = face_landmarks.landmark[landmark_idx]
                            landmark_x = int(landmark.x * frame.shape[1])
                            landmark_y = int(landmark.y * frame.shape[0])
                            row.extend([landmark_x, landmark_y])
                    
                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Write the row for the current landmark
                csv_writer.writerow(row)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete: {video_path}")

def process_video_wrapper(args):
    video_path, output_path, focus = args
    start_time = time.time()
    process_video(video_path, output_path, focus)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processing time for {video_path}: {total_time} seconds")

if __name__ == "__main__":
    # Face part arrays
    Lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    Left_Eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    Left_Eyebrow = [276, 283, 282, 295, 300, 293, 334, 296, 336]
    Right_Eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 33, 246, 161, 160, 159, 158, 157, 173, 133]
    Right_Eyebrow =  [33, 7, 163, 144, 145, 153, 154, 155, 133, 33, 246, 161, 160, 159, 158, 157, 173, 133]
    Outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    All = list(range(1,469))
    focus = [4] + Lips

    # Folder containing input videos
    input_folder = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\TrimmedVids')

    # Output folder for processed videos
    output_folder = "Dump"
    os.makedirs(output_folder, exist_ok=True)

    # Process the first 20 videos in the input folder
    video_files = [(os.path.join(input_folder, filename), os.path.join(output_folder, filename)) for filename in os.listdir(input_folder)[:20] if filename.endswith((".mp4", ".avi"))]

    # Timing
    start_time = time.time()

    # Set up multiprocessing pool
    print("Starting video processing...")
    pool = multiprocessing.Pool()
    pool.map(process_video_wrapper, video_files)
    pool.close()
    pool.join()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Video processing complete. Total time taken: {total_time} seconds.")
    print(f"Average time per video: {total_time / len(video_files)} seconds.")
