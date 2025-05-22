from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/Users/zhengjeppesen/Desktop/project/21Game/dataset_split/", epochs=100, imgsz=64)