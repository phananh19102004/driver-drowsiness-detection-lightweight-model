import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# data path
data_dir = r'G:\dataset_split'

# data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=2) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print('Classes:', class_names)

# Load ResNet-18 
model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)
model_ft = model_ft.to(device)

# Define loss and optimizer with L2 regularization (weight decay)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-5)

# Reduce learning rate if val_loss is not improve
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # tqdm 
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} [{epoch+1}/{num_epochs}]', leave=False)

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'val':
                scheduler.step(running_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_all.append(epoch_loss)
            else:
                val_loss_all.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            elif phase == 'val':
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_wts)
                return model

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

    model.load_state_dict(best_model_wts)

    plt.plot(train_loss_all, label='Train loss')
    plt.plot(val_loss_all, label='Val loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.show()

    return model

if __name__ == "__main__":
    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=15)
    torch.save(model_ft.state_dict(), 'drowsy_driver_resnet18.pth')
