import os
import cv2
import tkinter as tk
from tkinter import simpledialog

class VideoRecorder:
    def __init__(self, name, length):
        self.name = "yell2" + name  # Automatically prepend "yes_" to the video name
        self.length = length
        self.folder = "Train"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(os.path.join(self.folder, f"{self.name}.mp4"), fourcc, 20.0, (640, 480))
        self.record()

    def record(self):
        start_time = cv2.getTickCount()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.out.write(frame)
            cv2.imshow('Recording', frame)
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time >= self.length:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    root.withdraw()
    video_name = simpledialog.askstring("Video Name", "Enter the name of the video:")
    # Automatically set the video length to 4 seconds
    video_length = 4
    if video_name:
        recorder = VideoRecorder(video_name, video_length)
    else:
        print("Video name is required.")
        
if __name__ == "__main__":
    main()
