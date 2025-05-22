from ultralytics import YOLO
import cv2
import os

# 加载模型
model = YOLO('C:/Users/HP/Desktop/21/21Game-main/runs/classify/train/weights/best.pt')

# 加载图像并预测
image_path = "C:/Users/HP/Desktop/21/21Game-main/dataset_split/val/8/8c_1.png"
results = model(image_path)

# 绘制预测结果
annotated_frame = results[0].plot()

# 保存图像
output_path = "output_result.png"
cv2.imwrite(output_path, annotated_frame)
print(f"✅ 识别结果已保存为：{os.path.abspath(output_path)}")
