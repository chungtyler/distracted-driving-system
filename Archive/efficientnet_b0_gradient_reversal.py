import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

from torch.utils.data import Dataset

class DriverDataset(Dataset):
    def __init__(self, subset, driver_labels):
        self.subset = subset
        self.driver_labels = driver_labels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, action_label = self.subset[idx]
        driver_label = self.driver_labels[idx]
        return image, action_label, driver_label

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
        for images, action_labels, driver_labels in val_loader:
            images = images.to(device)
            action_labels = action_labels.to(device)
            driver_labels = driver_labels.to(device)

            # Forward pass: get both heads
            action_logits, driver_logits = model(images, alpha=0)

            # Compute loss only on the action head
            loss = criterion(action_logits, action_labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(action_logits, 1)
            correct += (predicted == action_labels).sum().item()
            total += action_labels.size(0)

    val_accuracy = correct / total
    val_loss = running_loss / len(val_loader)
    return val_loss, val_accuracy

# def val_model(model, val_loader, criterion, device):
#     # Validate model based on inference (no weight updates)
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, action_labels, driver_labels in val_loader:
#             images, action_labels, driver_labels = images.to(device), action_labels.to(device), action_labels.to(driver_labels)

#             outputs, _ = model(images, alpha=0) # Generate model predictions
#             loss = criterion(outputs, labels) # Generate loss

#             running_loss += loss.item()  # Calculate validation loss
            
#             _, predicted = torch.max(outputs, 1) # Predicted class
#             correct += (predicted == labels).sum().item() # Number of correct predictions
#             total += labels.size(0) # Total number of samples

#     val_accuracy = correct / total
#     val_loss = running_loss / len(val_loader)
#     return val_loss, val_accuracy

# def train_model(model, train_loader, optimizer, criterion, device, alpha):
#     # Train model and track running loss, training accuracy, and validation accuracy
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad() # Clear the previous gradients of .grad
#         outputs = model(images, alpha=alpha) # Generate model predictions
#         loss = criterion(outputs, labels) # Based on model predictions and labels generate loss
#         loss.backward() # Calculate weights (back propogation), stored in .grad 
#         optimizer.step() # Update weights

#         running_loss += loss.item() # Calculate running loss

#         _, predicted = torch.max(outputs, 1) # Predicted class
#         correct += (predicted == labels).sum().item() # Number of correct predictions
#         total += labels.size(0) # Total number of samples

#     train_loss = running_loss / len(train_loader)
#     train_accuracy = correct / total
#     return train_loss, train_accuracy

# def train_model(model, train_loader, optimizer, criterion, device, alpha, lambda_driver=0.5):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, action_labels, driver_labels in train_loader:
#         images = images.to(device)
#         action_labels = action_labels.to(device)
#         driver_labels = driver_labels.to(device)

#         optimizer.zero_grad()

#         # Forward pass: model returns both heads
#         action_logits, driver_logits = model(images, alpha=alpha)

#         # Compute losses
#         loss_action = criterion(action_logits, action_labels)
#         loss_driver = criterion(driver_logits, driver_labels)

#         # Total loss = action loss + λ * driver loss
#         loss = loss_action + lambda_driver * loss_driver

#         # Backpropagation
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         # Track training accuracy (action head only)
#         _, predicted = torch.max(action_logits, 1)
#         correct += (predicted == action_labels).sum().item()
#         total += action_labels.size(0)

#     train_loss = running_loss / len(train_loader)
#     train_accuracy = correct / total
#     return train_loss, train_accuracy

def train_model(model, train_loader, optimizer, criterion, device, alpha, lambda_driver=0.5):
    model.train()
    running_loss, running_action_loss, running_driver_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0

    for images, action_labels, driver_labels in train_loader:
        images = images.to(device)
        action_labels = action_labels.to(device)
        driver_labels = driver_labels.to(device)

        optimizer.zero_grad()
        action_logits, driver_logits = model(images, alpha=alpha)

        loss_action = criterion(action_logits, action_labels)
        loss_driver = criterion(driver_logits, driver_labels)
        loss = loss_action + lambda_driver * loss_driver

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_action_loss += loss_action.item()
        running_driver_loss += loss_driver.item()

        _, predicted = torch.max(action_logits, 1)
        correct += (predicted == action_labels).sum().item()
        total += action_labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_action_loss = running_action_loss / len(train_loader)
    train_driver_loss = running_driver_loss / len(train_loader)
    train_accuracy = correct / total

    return train_loss, train_action_loss, train_driver_loss, train_accuracy

def get_alpha(epoch, max_epoch, max_alpha=1.0):
    # Sigmoid ramp-up
    progress = epoch / max_epoch
    return max_alpha * (2.0 / (1.0 + torch.exp(-10 * torch.tensor(progress))) - 1.0)

# class EfficientNet:
#     def __init__(self, output_features):
#         self.weights = models.EfficientNet_B0_Weights.DEFAULT       # Pre-trained EfficientNet weights (B0 to B7)
#         self.transform = self.weights.transforms()                  # Convert input images to standard format
#         self.model = models.efficientnet_b0(weights=self.weights)   # Load EfficientNet model with Pre-trained weights

#         self.model.classifier = nn.Sequential(                                  # Replace linear layer to match classification problem
#             nn.Dropout(p=0.2, inplace=True),                                    # Reduce overfitting (drops 20% of neurons randomly)
#             nn.Linear(self.model.classifier[1].in_features, output_features)    # Replace linear layer (1280, 1000) to (1280, num_classes)
#         )

#     def load_weights(self, path):
#         self.model.load_state_dict(torch.load(path))

# Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # clone ensures it's not a view (fixes RuntimeError)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# Helper function
def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class EfficientNetInvariant(nn.Module):
    def __init__(self, num_actions, num_drivers):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.transform = weights.transforms()
        backbone = models.efficientnet_b0(weights=weights)

        # Remove the classifier, keep the feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.classifier[1].in_features  # 1280 for B0

        # Action classification head
        self.action_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feat_dim, num_actions)
        )

        # Driver classification head (adversarial)
        self.driver_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feat_dim, num_drivers)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # Action prediction
        action_logits = self.action_head(features)

        # Driver prediction with gradient reversal
        rev_features = grad_reverse(features, alpha)
        driver_logits = self.driver_head(rev_features)

        return action_logits, driver_logits

def main():
    '''LOAD DATA'''
    # Load EfficientNet Weights and Input Pre-Processing Settings
    weights = models.EfficientNet_B0_Weights.DEFAULT # Pre-trained EfficientNet weights (B0 to B7)
    transform = weights.transforms() # Convert input images to standard format

    # Load Dataset and Data Loaders
    dataset_path = 'C:/UWaterloo/Courses/ME 744 - Computational Intelligence/state-farm-distracted-driver-detection/imgs/train'
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Load the CSV file
    driver_image_list = pd.read_csv('C:/UWaterloo/Courses/ME 744 - Computational Intelligence/state-farm-distracted-driver-detection/driver_imgs_list.csv')
    driver_image_list["img"] = driver_image_list["img"].astype(str) # Convert to string file

    # Get each folder filename
    paths = [path for path, _ in dataset.samples]
    basenames = [os.path.basename(path) for path in paths]

    # Match each image with driver (subject number)
    map_driver_image_list = pd.DataFrame({"img": basenames}).merge(driver_image_list[["img", "subject"]], on="img", how="left")

    if map_driver_image_list["subject"].isna().any():
        # If this triggers, your DATA_ROOT or CSV_PATH is off, or filenames differ
        missing = map_driver_image_list[map_driver_image_list["subject"].isna()].head(5)
        raise RuntimeError(f"Images not found in CSV (check paths). Examples:\n{missing}")

    groups = map_driver_image_list["subject"].values  # one group per sample (driver id)

    # Split so driver appears in both train and val
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_indices, val_indices = next(gss.split(basenames, groups=groups))

    # train_dataset = Subset(dataset, train_indices) # Load training dataset (80%)
    # val_dataset = Subset(dataset, val_indices) # Load validate dataset (20%)

    # Map driver IDs to integers
    driver_to_idx = {driver: i for i, driver in enumerate(map_driver_image_list["subject"].unique())}
    driver_labels = map_driver_image_list["subject"].map(driver_to_idx).values

    # Wrap subsets so they yield (image, action_label, driver_label)
    train_dataset = DriverDataset(Subset(dataset, train_indices), driver_labels[train_indices])
    val_dataset   = DriverDataset(Subset(dataset, val_indices), driver_labels[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # Create training data loader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # Create validation data loader

    '''LOAD MODEL'''
    # Load and Edit Model
    # efficientnet_b0 = EfficientNetInvariant(len(dataset.classes))
    # # efficientnet_b0.load_weights('efficientnet_b0.pth')
    # model = efficientnet_b0.model
    model = EfficientNetInvariant(num_actions=len(dataset.classes), num_drivers=len(np.unique(groups)))


    '''TRAIN MODEL AND VALIDATE'''
    # Model Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cuda cores or CPU cores
    model.to(device) # Load model onto device

    print(f"Model loaded {device}")
    
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

    num_epochs = 30 # Number of epochs

    best_val_accuracy = 0.0

    # Plotting parameters
    epochs = []
    train_losses = []
    val_losses = []
    _, ax = plt.subplots()

    print("Training Starting...")

    # Train model and validate for each epoch updating the best model to save
    for epoch in range(num_epochs):
        alpha = get_alpha(epoch, num_epochs, max_alpha=1.0)
        train_loss, train_action_loss, train_driver_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device, alpha)
        val_loss, val_accuracy = val_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} || Training Action Loss: {train_action_loss} || Training Driver Loss: {train_driver_loss} || Training Accuracy: {train_accuracy} || Validation Loss: {val_loss} || Validation Accuracy: {val_accuracy}")

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

    # Plot Confusion Matrix
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

    # Classification report (precision, recall, f1-score, support, accuracy, macro average, weighted average)
    classification_report = metrics.classification_report(y_true, y_predicted, digits=3)
    print(f"Classification Report:\n{classification_report}")

    # Top-K accuracy
    top_3_accuracy = metrics.top_k_accuracy_score(y_true, y_probabilities, k=3)
    print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")

    plt.ioff()
    plt.show()

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
