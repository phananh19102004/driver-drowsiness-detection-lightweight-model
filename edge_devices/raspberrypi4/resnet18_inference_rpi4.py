import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import time
import os


model_path = '/home/pi/Downloads/modelalgindata.pth'  
test_dir = '/home/pi/Downloads/test'                  
device = torch.device("cpu")               

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

# === Load model ===
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, len(class_names))
)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Inference ===
total_time = 0
all_preds = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)

        start_time = time.time()
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        end_time = time.time()

        elapsed = end_time - start_time
        total_time += elapsed

        img_path, _ = dataset.samples[i]
        pred_class = class_names[pred.item()]
        print(f" {os.path.basename(img_path)} ? Predicted: {pred_class} | {elapsed:.4f}s")


num_images = len(dataset)
avg_time = total_time / num_images
fps = 1 / avg_time if avg_time > 0 else 0

print(f"\n Total IMG: {num_images}")
print(f" inferences time: {avg_time:.4f} second")
print(f" FPS : {fps:.2f}")
