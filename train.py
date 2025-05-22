from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/Users/zhengjeppesen/Desktop/project/21Game/dataset_split/", epochs=100, imgsz=64)

# Validate the model
val_results = model.val(data="/Users/zhengjeppesen/Desktop/project/21Game/dataset_split/")

# Save the model
model.save("best.pt")

# Print validation results
print("\nValidation Results:")
print(f"Accuracy: {val_results.top1:.2f}%")
print(f"Top-5 Accuracy: {val_results.top5:.2f}%")