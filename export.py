from ultralytics import YOLO


model = YOLO('/home/orin/ultralytics_old/runs/detect/egg3/weights/best.pt')

# Export the model
model.export(format='engine', half= True)#, dynamic=True, batch=128)


#try versions on orion
