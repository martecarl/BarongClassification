from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('C:\\Users\\Carl\\Barong Tagalog Design Classification\\yolov8scls.pt')

# Export to ONNX
model.export(format='onnx', dynamic=True)  # 'dynamic=True' allows for dynamic axes for batch size
