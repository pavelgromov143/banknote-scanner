from ultralytics import YOLO

model = YOLO('yolov8n.pt')
result = model.train(data = './data/data.yaml', epochs = 50, imgsz = 640 )
