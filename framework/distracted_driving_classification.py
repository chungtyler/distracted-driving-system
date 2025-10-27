import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os

# Load EfficientNet Weights and Input Pre-Processing Settings
weights = models.EfficientNet_B0_Weights.DEFAULT # Pre-trained EfficientNet weights (B0 to B7)
transform = weights.transforms() # Convert input images to standard format
# transform = transforms.Compose([
#     transforms.Resize(256),      # Resize image to 256 px (shorter length)
#     transforms.CenterCrop(224),  # Crop the image 224 x 224 px
#     transforms.ToTensor(),       # Conver to Pytorch Tensor
#     transforms.Normalize(        # Normalize based on image pixel distribution
#         mean=[0.485, 0.456, 0.406], # Calculate your own (this is from ImageNet)
#         std=[0.229, 0.224, 0.225] # Calculate your own (this is from ImageNet)
#     )
# ])

# Load Dataset and Data Loaders
dataset_path = ''
train_dataset = datasets.ImageFolder(root='root/img/train', transform=transform) # Load training dataset (80%)

#TODO convert training dataset and grab 20% from each category and use for validation
val_dataset = datasets.ImageFolder(root='root/img/val', transform=transform) # Load validate dataset (20%)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # Create training data loader
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True) # Create validation data loader

# Load and Edit Model
model = models.efficientnet_b0(weights=weights) # Load EfficientNet model with Pre-trained weights
num_classes = len(train_dataset.classes) # Get the number of classification classes
model.classifier = nn.Sequential(                               # Replace linear layer to match classification problem
    nn.Dropout(p=0.2, inplace=True),                            # Reduce overfitting (drops 20% of neurons randomly)
    nn.Linear(model.classifier[1].in_features, num_classes)     # Replace linear layer (1280, 1000) to (1280, num_classes)
)

# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cuda cores or CPU cores
model.to(device) # Load model onto device

criterion = torch.nn.CrossEntropyLoss() # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Optimizer

num_epochs = 10 # Number of epochs

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader():
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Model Evaluation

# (Optional) Fine-Tuning
