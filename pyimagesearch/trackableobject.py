class TrackableObject:
	def __init__(self, objectID, centroid, pred_label):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		self.pred_label_history = [pred_label]
