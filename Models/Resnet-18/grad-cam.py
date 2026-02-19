import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# -------- CONFIG --------
MODEL_PATH = "modelcropdata.pth"
IMG_PATH = "drowsy2.jpg"
DATASET_PATH = "G:/dataset_cropped/train"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load class names 
class_names = datasets.ImageFolder(DATASET_PATH).classes
print("Class names:", class_names)

# Init model 
def get_model(num_classes: int):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# data_preprocessing
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0).to(DEVICE), img  

# Grad-CAM
def generate_gradcam(model, input_tensor, target_layer):
    features = []
    gradients = []

    def forward_hook(module, inp, out):
        features.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # hook
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    model.eval()

    output = model(input_tensor)                
    pred_class = int(output.argmax(dim=1).item())

    class_score = output[0, pred_class]
    class_score.backward()

    #  fmap & grad
    fmap = features[0].detach().cpu().numpy()[0]        # (C,H,W)
    grads_val = gradients[0].detach().cpu().numpy()[0]  # (C,H,W)

    # weights & cam
    weights = np.mean(grads_val, axis=(1, 2))           # (C,)
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)    # (H,W)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)                            # ReLU
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    # normal
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-12:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)

    fh.remove()
    bh.remove()

    return cam, pred_class

# Overlay heatmap 
def overlay_cam_on_image(img_pil, cam, alpha=0.4):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
    img_np = np.array(img_pil.resize((224, 224)))  
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    superimposed_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
    return superimposed_rgb, superimposed_bgr  


if __name__ == "__main__":
    model = get_model(num_classes=len(class_names)).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    input_tensor, raw_image = preprocess_image(IMG_PATH)
    
    target_layer = model.layer4[-1]  

    cam, pred_class = generate_gradcam(model, input_tensor, target_layer)
    pred_label = class_names[pred_class]

    vis_rgb, vis_bgr = overlay_cam_on_image(raw_image, cam, alpha=0.4)

    
    plt.imshow(vis_rgb)
    plt.title(f"Grad-CAM - Predicted: {pred_label} ({pred_class})")
    plt.axis('off')
    plt.show()

    out_path = f"gradcam_{pred_label}.jpg"
    cv2.imwrite(out_path, vis_bgr)
    print("Saved:", os.path.abspath(out_path))
