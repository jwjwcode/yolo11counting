from ultralytics import YOLO

# Load a model
model = YOLO('/home/nx/ultralytics/runs/detect/rotifer10/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True, device='0')
