import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# === CONFIG ===
model_path = r'C:\Users\Hi\Desktop\model.pth'  
data_dir = r'G:\dataset_split'                 
batch_size = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(os.path.join(data_dir, 'val')):
    raise FileNotFoundError(f"Validation folder not found: {os.path.join(data_dir, 'val')}")

# Data transforms 
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load validation dataset 
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = val_dataset.classes
print("Classes:", class_names)

# Load model 
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 2)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def evaluate_model_metrics(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_time = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start = time.time()
            outputs = model(inputs)
            end = time.time()

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            total_time += (end - start)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    avg_time = total_time / len(all_labels)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print("\n Evaluation Metrics:")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-score:       {f1:.4f}")
    print(f"AUC (ROC Area): {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Inference Time: {avg_time * 1000:.2f} ms/image")
    print(f"FPS:            {fps:.2f}")

    # Plot ROC 
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

evaluate_model_metrics(model, val_loader, class_names, device)
