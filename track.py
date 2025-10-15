import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/nx/ultralytics/runs/detect/rotifer28/weights/best.engine')

#results = model.track(source="/home/nx/datasets/speed1live2.mp4", imgsz=320, show=True)  # Tracking with default tracker

# Open the video file
video_path = "/home/nx/datasets/speed1live2.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, imgsz=320)
        print('results', results)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
        	break
    else:
    	break

