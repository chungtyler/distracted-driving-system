import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
from sklearn import metrics
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses(ax, epoch, train_losses, val_losses):
    ax.clear()
    ax.plot(epoch, val_losses, label='Validation Loss', color='blue', linestyle='-')
    ax.plot(epoch, train_losses, label='Training Loss', color='orange', linestyle='-')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Epoch')
    ax.legend()
    ax.grid(True)

def plot_confusion_matrix(y_true, y_predicted, class_names):
    confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.title('Distracted Driving Classification Confusion Matrix')
    plt.tight_layout()

def val_model(model, val_loader, criterion, device):
    # Validate model based on inference (no weight updates)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # Generate model predictions
            loss = criterion(outputs, labels) # Generate loss

            running_loss += loss.item()  # Calculate validation loss
            
            _, predicted = torch.max(outputs, 1) # Predicted class
            correct += (predicted == labels).sum().item() # Number of correct predictions
            total += labels.size(0) # Total number of samples

    val_accuracy = correct / total
    val_loss = running_loss / len(val_loader)
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

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def main():
    '''LOAD DATA'''
    # Load EfficientNet Weights and Input Pre-Processing Settings
    weights = models.EfficientNet_B0_Weights.DEFAULT # Pre-trained EfficientNet weights (B0 to B7)
    transform = weights.transforms() # Convert input images to standard format

    # Load Dataset and Data Loaders
    dataset_path = 'C:/UWaterloo/Courses/ME 744 - Computational Intelligence/state-farm-distracted-driver-detection/imgs/train'
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

    # LOAD MODEL

    '''TRAIN MODEL AND VALIDATE'''
    # Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cuda cores or CPU cores
    model.to(device) # Load model onto device

    print(f"Model loaded {device}")
    
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

    num_epochs = 1 # Number of epochs

    best_val_accuracy = 0.0

    # Plotting parameters
    epochs = []
    train_losses = []
    val_losses = []
    _, ax = plt.subplots()

    print("Training Starting...")

    # Train model and validate for each epoch updating the best model to save
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = val_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} || Training Loss: {train_loss} || Training Accuracy: {train_accuracy} || Validation Loss: {val_loss} || Validation Accuracy: {val_accuracy}")

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "efficientnet_distracted_driving.pth")
            print("Saved new best model!")

        # Plot Loss vs Epoch Curve
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(ax, epochs, train_losses, val_losses)
        plt.pause(0.1)

    # Plot ROC and Confusion Matrix
    model.eval()
    y_true = []
    y_predicted = []
    y_probabilities = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Predicted class
            probabilities = torch.softmax(outputs, 1) # Predicted class distribution

            y_true.extend(labels.cpu().numpy())
            y_predicted.extend(predicted.cpu().numpy())
            y_probabilities.extend(probabilities.cpu().numpy())

    plot_confusion_matrix(y_true, y_predicted, dataset.classes)

    plt.ioff()
    plt.show()

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
