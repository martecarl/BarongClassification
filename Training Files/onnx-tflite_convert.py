from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("yolov8-cls.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("yolov8-cls.pb")
