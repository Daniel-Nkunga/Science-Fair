import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip, ImageSequenceClip

def crop_video_around_point(video_path, landmark_index):
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)

    # Load the video clip
    clip = VideoFileClip(video_path)

    # Initialize variables for cropping dimensions
    crop_width = 300
    crop_height = 300

    # Initialize the output video frames
    processed_frames = []

    # Process each frame of the video
    for frame in clip.iter_frames(fps=20, dtype='uint8'):
        # Convert the frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        results = holistic.process(rgb_frame)

        # Retrieve the coordinates of the specified landmark
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_index]
            point_x = int(landmark.x * frame.shape[1])
            point_y = int(landmark.y * frame.shape[0])

            # Calculate cropping dimensions
            crop_x1 = max(point_x - crop_width // 2, 0)
            crop_x2 = min(point_x + crop_width // 2, frame.shape[1])
            crop_y1 = max(point_y - crop_height // 2, 0)
            crop_y2 = min(point_y + crop_height // 2, frame.shape[0])

            # Crop the frame
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            # print("chip chop")

            # Resize the cropped frame to a square with 300 length sides
            resized_cropped_frame = cv2.resize(cropped_frame, (300, 300))

            # Append the processed frame to the output list
            processed_frames.append(resized_cropped_frame)
    print("Processed Frames:", len(processed_frames))  # Debug print to see how many frames are processed

    # Release MediaPipe resources
    holistic.close()

    # Create a video clip from processed frames
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    print("Done")
    return processed_clip

# Example usage:
input_video_path = "input.mp4"  # Change this to the path of your input video
output_video_path = "output.mp4"  # Change this to the desired output video path
landmark_index = 4  # Index of the landmark to track

cropped_video = crop_video_around_point(input_video_path, landmark_index)
cropped_video.write_videofile(output_video_path, codec="libx264", fps=20)  # Adjust codec and fps as needed
