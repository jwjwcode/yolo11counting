# import the necessary packages
import numpy as np
np.bool = np.bool_
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import cv2
from ultralytics import YOLO

import sys
import time

	

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--modelpath", default = 'best.pt',
	help="path to detection model")
ap.add_argument("--input", type=str, default = "rotifer_stage.avi",
	help="path to optional input video file")
ap.add_argument("--output", type=str, default='count_result_docker.mp4',
	help="path to optional output video file")
ap.add_argument("--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("--skip-frames", type=int, default=1,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
predictor = YOLO(args['modelpath'])
	
# initialize the frame dimensions 
I_SZ = 320
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
total_E = 0
total_N = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")#(*"MJPG")

vs = cv2.VideoCapture(args["input"])
success, frame = vs.read()
if not success:
    print("[ERROR] Couldn't read the first frame.")
    sys.exit(1)

(H, W) = frame.shape[:2]
writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
Total = 0
tic = time.time()
# loop over frames from the video stream

while vs.isOpened():
	Total += 1
	if Total > 100:
	    break
	success, frame = vs.read()
	if not success:
		break
	results = predictor.track(source=frame,tracker='custom_tracker.yaml', persist=True, conf=0.3001, iou=0.5, imgsz=I_SZ,agnostic_nms=True)
	boxes, labels, probs, objectIDs = results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.id	
	cv2.line(frame, (int(W*0.5),0), (int(W*0.5), H), (0, 255, 255), 2)
	# loop over the tracked objects
	if objectIDs is not None:
		for j in range(labels.shape[0]):
        #draw bounding boxes
			box = boxes[j, :]
			if labels[j] == 0:	
				cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)           		
			elif labels[j] == 1:
				cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1) 			
			cX = int((boxes[j][0] + boxes[j][2]) / 2.0)
			cY = int((boxes[j][1] + boxes[j][3]) / 2.0)
			centroid = (cX,cY)
			pred_label = labels[j]
		
			objectID = objectIDs[j].item()
		# check to see if a trackable object exists for the current

			to = trackableObjects.get(objectID, None)
			if to is None:
				to = TrackableObject(objectID, centroid, pred_label)
		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
			else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
				y = [c[0] for c in to.centroids]
				avg_label = sum(to.pred_label_history) / len(to.pred_label_history)
				direction = centroid[0] - np.mean(y)
				to.centroids.append(centroid)
				to.pred_label_history.append(pred_label)

			# check to see if the object has been counted or not
				if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
					if (direction > 0 and centroid[0] > int(W*0.5)) or (direction < 0 and centroid[0] < int(W*0.5)):
						if avg_label < 0.5: 
							total_E += 1
							to.counted = True
						elif avg_label >= 0.5:
							total_N += 1
							to.counted = True
		# store the trackable object in our dictionary
			trackableObjects[objectID] = to
        
	info = [
		("E", total_E),
		("N", total_N),
		]
	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	writer.write(frame)
	# show the output frame	
	#cv2.imshow("Frame", frame)
	#key = cv2.waitKey(1) & 0xFF   
	#if key == ord("q"):
	#	break
cv2.imwrite('a.png', frame)
print('total N {}, total E {}'.format(total_N,total_E))
toc = time.time()
total_time = toc - tic
fps = Total / total_time
print('time {}, fps {}'.format(total_time, fps))
writer.release()	
# close any open windows
#cv2.destroyAllWindows()
		


