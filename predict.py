from ultralytics import YOLO
import cv2

model = YOLO('/home/nx/ultralytics/runs/detect/rotifer28/weights/best.engine')
video_path = "/home/nx/datasets/speed1live2.mp4"
cap = cv2.VideoCapture(video_path)

c=0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,imgsz=320,agnostic_nms=True)
        print('xyxy', results[0].boxes.xyxy)
        print('cls', results[0].boxes.cls)
        print('conf', results[0].boxes.conf)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        c+=1
        if c>10000:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
