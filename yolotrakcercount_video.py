# import the necessary packages
import numpy as np
np.bool = np.bool_
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import cv2
from ultralytics import YOLO
import PySpin
import sys
import time


def Prepare_camera(cam):
	
	print('cam list', len(cam_list))
	processor = PySpin.ImageProcessor()
	processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
	
        # Retrieve TL device nodemap and print device information
	nodemap_tldevice = cam.GetTLDeviceNodeMap()
        # Initialize camera
	cam.Init()
        # Retrieve GenICam nodemap
	nodemap = cam.GetNodeMap()
	node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
	if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
		print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        # Retrieve entry node from enumeration node
	node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
	if not PySpin.IsReadable(node_acquisition_mode_continuous):
		print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        # Retrieve integer value from entry node
	acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        # Set integer value from entry node as new value of enumeration node
	node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
	print('Acquisition mode set to continuous...')
	
	# Retrieve Stream Parameters device nodemap
	s_node_map = cam.GetTLStreamNodeMap()
	# Retrieve Buffer Handling Mode Information
	handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
	if not PySpin.IsReadable(handling_mode) or not PySpin.IsWritable(handling_mode):
		print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
		return False

	handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
	if not PySpin.IsReadable(handling_mode_entry):
		print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
		return False            
	print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        
        #set the buffer handling mode
	handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
	handling_mode.SetIntValue(handling_mode_entry.GetValue())
	print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
	#start acquisition	
	cam.BeginAcquisition()
	device_serial_number = ''
	node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
	if PySpin.IsReadable(node_device_serial_number):
		device_serial_number = node_device_serial_number.GetValue()
		print('Device serial number retrieved as %s...' % device_serial_number)
	processor = PySpin.ImageProcessor()

        # Set default image processor color processing method
        #
        # *** NOTES ***
        # By default, if no specific color processing algorithm is set, the image
        # processor will default to NEAREST_NEIGHBOR method.
	processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
	node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
	framerate_to_set = node_acquisition_framerate.GetValue()
	print('Frame rate to be set to %d...' % framerate_to_set)
	
	return cam, processor, framerate_to_set	

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--modelpath", default = 'best.engine',
	help="path to detection model")
ap.add_argument("--input", type=str, default = "rotifer_stage.avi",
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
	
# initialize the frame dimensions 
W = 640
H = 480
I_SZ = 320
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
total_E = 0
total_N = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")#(*"MJPG")
writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(720, 540), True)
vs = cv2.VideoCapture(args["input"])
Total = 0
tic = time.time()
# loop over frames from the video stream

while vs.isOpened():
	Total += 1
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
	#writer.write(frame)
	# show the output frame	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF   
	if key == ord("q"):
		break

print('total N {}, total E {}'.format(total_N,total_E))
toc = time.time()
total_time = toc - tic
fps = Total / total_time
print('time {}, fps {}'.format(total_time, fps))
#writer.release()	
# close any open windows
cv2.destroyAllWindows()
		


