
import onnx

# Load the ONNX model
onnx_model = onnx.load('yolov8n-cls.onnx')

# Check the model for correctness
onnx.checker.check_model(onnx_model)

print("ONNX model is valid!")
