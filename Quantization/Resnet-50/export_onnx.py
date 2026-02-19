import torch
from torchvision import models
import torch.nn as nn

model_path = "drowsy_driver_resnet50.pth"
onnx_path = "drowsy_driver_resnet50.onnx"
num_classes = 2

device = "cpu"

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["logits"],
    opset_version=12,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
)

print("Exported:", onnx_path)
