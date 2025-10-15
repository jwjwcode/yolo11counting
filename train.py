from ultralytics import YOLO

#load the model.
model = YOLO('yolov8s.pt')

#training

results = model.train(
                      data = 'egg.yaml',
                      imgsz=320,
                      epochs=20,
                      batch=64,
                      name='egg'
                      )
