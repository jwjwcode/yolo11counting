# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import cv2
from ultralytics import YOLO
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--modelpath", default = '/home/orin/ultralytics_old/runs/detect/rotifer_n_320/weights/best.engine', #egg3
	help="path to detection model")
ap.add_argument("--input", type=str, default = "/home/orin/datasets/rotifer/speed1live2.mp4", #egg/eggcount2
	help="path to optional input video file")
ap.add_argument("--output", type=str, default='count_result.mp4',
	help="path to optional output video file")
ap.add_argument("--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("--skip-frames", type=int, default=1,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
predictor = YOLO(args['modelpath'])
print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["input"])
	
# initialize the video writer (we'll instantiate later if need be)
writer = None
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
I_SZ = 320
if args["output"] is not None and writer is None:
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(720, 540), True)
# instantiate our centroid tracker, then initialize a list to store
# each of our trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=90) # (40, 90) for low speed; (10, 200) for high speed
trackers = []
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
total_E = 0
total_N = 0

tic = time.time()
# loop over frames from the video stream
while vs.isOpened():
	success, frame = vs.read()
	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	rects = []
	results = predictor.predict(frame,imgsz=I_SZ,agnostic_nms=True)
	boxes, labels, probs = results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
		# loop over the detections
	for i in range(0, boxes.size(0)):
        #draw bounding boxes
		box = boxes[i, :]
		if labels[i] == 0:	
			cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)           		
		elif labels[i] == 1:
			cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1) 
		confidence = probs[i]
			# filter out weak detections by requiring a minimum
			# confidence
		if confidence > args["confidence"]:
			(startX, startY, endX, endY) = boxes[i].cpu().detach().numpy().astype("int")
			rects.append((startX, startY, endX, endY))
	# draw a line for counting
	cv2.line(frame, (int(W*0.7),0), (int(W*0.7), H), (0, 255, 255), 2)
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects, labels, W)
	
	# loop over the tracked objects
	for (objectID, v) in objects.items():
		centroid = v[0]
		pred_label = v[1]
		box_rect = v[2]

		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)
		# if there is no existing trackable object, create one
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
				if direction > 0 and centroid[0] > int(W*0.7):
					if avg_label < 0.5: 
						total_E += 1
						to.counted = True
					elif avg_label >= 0.5:
						total_N += 1
						to.counted = True
		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("E", total_E),
		("N", total_N),
	]
	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# check to see if we should write the frame to disk
	#if writer is not None:
		#writer.write(frame)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# increment the total number of frames processed thus far and
	totalFrames += 1
	if not success:
		break
toc = time.time()
total_time = toc - tic
fps = totalFrames / total_time
print('time {}, fps {}'.format(total_time, fps))

# check to see if we need to release the video writer pointer
#if writer is not None:
	#writer.release()
vs.release()
# close any open windows
cv2.destroyAllWindows()
		


