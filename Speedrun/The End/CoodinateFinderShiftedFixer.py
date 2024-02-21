import cv2
import os
import mediapipe as mp
import csv

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to calculate displacement
def calculate_displacement(landmark, target_x, target_y):
    displacement_x = target_x - landmark.x
    displacement_y = target_y - landmark.y
    return displacement_x, displacement_y

# Function to process a single video
def process_video(video_path, output_path, focus):
    print(f"Processing video: {video_path}")
    # Initialize VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    # CSV file to store the coordinates
    csv_file_path = os.path.splitext(output_path)[0] + '.csv'
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
                            # Calculate displacement for landmark 4
                            target_x = 150 / frame.shape[1]
                            target_y = 150 / frame.shape[0]
                            landmark4 = face_landmarks.landmark[4]
                            displacement_x, displacement_y = calculate_displacement(landmark4, target_x, target_y)
                            
                            # Record landmark coordinates for the current frame
                            landmark = face_landmarks.landmark[landmark_idx]
                            landmark_x = int((landmark.x + displacement_x) * frame.shape[1])
                            landmark_y = int((landmark.y + displacement_y) * frame.shape[0])
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

if __name__ == "__main__":
    videos_to_process = ["nice2.mp4", "nice4.mp4", "nice5.mp4", "nice7.mp4", "nice8.mp4", "nice10.mp4", "nice11.mp4", "nice13.mp4", "nice14.mp4", "nice16.mp4", "nice17.mp4", "nice19.mp4", "nice21.mp4", "nice22.mp4", "nice24.mp4", "nice25.mp4", "no1.mp4", "no3.mp4", "no4.mp4", "no5.mp4", "no6.mp4", "no7.mp4", "no9.mp4", "no10.mp4", "no12.mp4", "no13.mp4", "no15.mp4", "no16.mp4", "no18.mp4", "no19.mp4", "no20.mp4", "no21.mp4", "no23.mp4", "no24.mp4", "no26.mp4", "no27.mp4", "no29.mp4", "no31.mp4", "no32.mp4", "no34.mp4", "no35.mp4", "no37.mp4", "no38.mp4", "no40.mp4", "no42.mp4", "no43.mp4", "no45.mp4", "no46.mp4", "no47.mp4", "no49.mp4", "yell1.mp4", "yell2.mp4", "yell3.mp4", "yell5.mp4", "yell6.mp4", "yell8.mp4", "yell9.mp4", "yell11.mp4", "yell12.mp4", "yell14.mp4", "yell15.mp4", "yell17.mp4", "yell18.mp4", "yell20.mp4", "yell22.mp4", "yell23.mp4", "yell25.mp4", "yes2.mp4", "yes3.mp4", "yes5.mp4", "yes6.mp4", "yes7.mp4", "yes9.mp4", "yes10.mp4", "yes11.mp4", "yes13.mp4", "yes14.mp4", "yes16.mp4", "yes17.mp4", "yes19.mp4", "yes21.mp4", "yes22.mp4", "yes24.mp4", "yes25.mp4", "yes27.mp4", "yes28.mp4", "yes30.mp4", "yes32.mp4", "yes33.mp4", "yes35.mp4", "yes36.mp4", "yes38.mp4", "yes39.mp4", "yes40.mp4", "yes41.mp4", "yes43.mp4", "yes44.mp4", "yes46.mp4", "yes47.mp4", "yes49.mp4"]

    # Face part arrays
    Lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    Left_Eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    Left_Eyebrow = [276, 283, 282, 295, 300, 293, 334, 296, 336]
    Right_Eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 33, 246, 161, 160, 159, 158, 157, 173, 133]
    Right_Eyebrow =  [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    Outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    All = list(range(1,469))
    focus = All

    for video_name in videos_to_process:
        input_video = os.path.join(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\TrimmedVids', video_name)
        output_file = os.path.join(r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Output', os.path.splitext(video_name)[0] + '_output')
        process_video(input_video, output_file, focus)

    # Process the specified video
    process_video(input_video, output_file, focus)
