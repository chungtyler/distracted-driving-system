import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
import multiprocessing
import time

def val_model(model, val_loader, criterion, device):
    # Validate model based on inference (no weight updates)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # Generate model predictions
            loss = criterion(outputs, labels) # Generate loss

            val_loss += loss.item()  # Calculate validation loss
            
            _, predicted = torch.max(outputs, 1) # Predicted class
            val_correct += (predicted == labels).sum().item() # Number of correct predictions
            val_total += labels.size(0) # Total number of samples

    val_accuracy = val_correct / val_total
    return val_loss, val_accuracy
    

def train_model(model, train_loader, optimizer, criterion, device):
    # Train model and track running loss, training accuracy, and validation accuracy
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Clear the previous gradients of .grad
        outputs = model(images) # Generate model predictions
        loss = criterion(outputs, labels) # Based on model predictions and labels generate loss
        loss.backward() # Calculate weights (back propogation), stored in .grad 
        optimizer.step() # Update weights

        running_loss += loss.item() # Calculate running loss

        _, predicted = torch.max(outputs, 1) # Predicted class
        correct += (predicted == labels).sum().item() # Number of correct predictions
        total += labels.size(0) # Total number of samples

    train_accuracy = correct / total
    
    return running_loss, train_accuracy

def main():
    '''LOAD DATA'''
    # Load EfficientNet Weights and Input Pre-Processing Settings
    weights = models.EfficientNet_B0_Weights.DEFAULT # Pre-trained EfficientNet weights (B0 to B7)
    transform = weights.transforms() # Convert input images to standard format

    # Load Dataset and Data Loaders
    dataset_path = 'state-farm-distracted-driver-detection/imgs/train'
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    y = dataset.targets

    train_indices, val_indices = train_test_split( # Split training dataset 80/20
        list(range(len(y))),
        stratify=y,
        test_size=0.2,
        random_state=50
    )

    train_dataset = Subset(dataset, train_indices) # Load training dataset (80%)
    val_dataset = Subset(dataset, val_indices) # Load validate dataset (20%)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # Create training data loader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # Create validation data loader

    '''LOAD MODEL'''
    # Load and Edit Model
    model = models.efficientnet_b0(weights=weights) # Load EfficientNet model with Pre-trained weights
    num_classes = len(dataset.classes) # Get the number of classification classes
    model.classifier = nn.Sequential(                               # Replace linear layer to match classification problem
        nn.Dropout(p=0.2, inplace=True),                            # Reduce overfitting (drops 20% of neurons randomly)
        nn.Linear(model.classifier[1].in_features, num_classes)     # Replace linear layer (1280, 1000) to (1280, num_classes)
    )

    '''TRAIN MODEL AND VALIDATE'''
    # Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cuda cores or CPU cores
    model.to(device) # Load model onto device

    print(f"Model loaded {device}")
    
    criterion = torch.nn.CrossEntropyLoss() # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Optimizer

    num_epochs = 20 # Number of epochs

    best_val_accuracy = 0.0

    print("Training Starting...")

    # Train model and validate for each epoch updating the best model to save
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = val_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} || Running Loss: {running_loss} || Training Accuracy: {train_accuracy} || Validation Loss: {val_loss} || Validation Accuracy: {val_accuracy}")

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "efficientnet_distracted_driving.pth")
            print("Saved new best model!")

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
