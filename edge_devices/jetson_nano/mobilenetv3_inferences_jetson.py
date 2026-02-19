import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import time
import os
import torch.nn as nn

model_path = '/home/phananh1910/Downloads/drowsy_driver_mobilenetv3.pth'
test_dir   = '/home/phananh1910/Downloads/test'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# === Data Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

dataset = ImageFolder(test_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
class_names = dataset.classes
print("Classes:", class_names)
print("Total IMG:", len(dataset))

# === Load model ===
model = models.mobilenet_v3_large(pretrained=False)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(class_names))

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Warmup ===
with torch.no_grad():
    it = iter(dataloader)
    for _ in range(100):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dataloader)
            x, _ = next(it)
        x = x.to(device)
        _ = model(x)
if device.type == "cuda":
    torch.cuda.synchronize()

# === Inference (timed - CUDA sync) ===
total_time = 0.0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        total_time += elapsed

        img_path, _ = dataset.samples[i]
        pred_class = class_names[pred.item()]
        print(f"{os.path.basename(img_path)} → Predicted: {pred_class} | {elapsed:.6f}s")

# === FPS ===
num_images = len(dataset)
avg_time = total_time / num_images if num_images > 0 else 0
fps = 1.0 / avg_time if avg_time > 0 else 0

print("\n==============================")
print(f"Total IMG: {num_images}")
print(f"Avg inference time: {avg_time:.6f} s/image")
print(f"FPS: {fps:.2f}")
print("==============================")
