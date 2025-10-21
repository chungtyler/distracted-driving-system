import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 5
DEVICE = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

# Load Datasets
transform = transforms.Compose([ # Converts images to PyTorch tensors and Normalize
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__() # Initialize NN module class
        '''
        nn.Conv2d(
            in_channels: colour channels (3 = RGB) or feature maps,
            out_channels: feature maps,
            kernel_size: filter/kernel nxn pixels,
            stride: number of pixels the kernel moves for input scanning,
            padding: adds extra pixels around image to prevent shrinkage)
        '''
        self.conv1 = nn.Conv2d(1, 32, 3) # 1 input, 32 outputs, 3x3 kernel
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3) # 32 inputs, 64 outputs, 3x3 kernel
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 128) # 5x5 size image with 64 channels, output 128
        self.fc2 = nn.Linear(128, 10) # Input 128, output 10 classes (0-9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) # First convolution and pooling layer
        x = self.pool2(F.relu(self.conv2(x))) # Second convolution and pooling layer
        x = x.view(-1, 64*5*5) # Flatten feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer Initialization
model = SimpleCNN().to(DEVICE) # Moves to CPU or GPU
print(model)
criterion = nn.CrossEntropyLoss() # Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Gradient Descent

# Training
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

for epoch in range(1, NUM_EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

MODEL_SAVE_PATH = "mnist_cnn.pt"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")

# MODEL_SAVE_PATH = "mnist_cnn.pt"
# loaded_model = SimpleCNN().to(DEVICE)
# loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
# loaded_model.eval()

# num_tests = 100
# indices = range(num_tests)
# subset_dataset = Subset(test_dataset, indices)

# # Create a DataLoader for the subset
# subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # 5. Evaluate the model on the 100 images
# test_loss = 0
# correct = 0
# elasped_time = 0

# print(f"Evaluating model on {len(subset_dataset)} images...")

# with torch.no_grad(): # Disable gradient calculations
#     for data, target in subset_loader:
#         data, target = data.to(DEVICE), target.to(DEVICE)
#         start_time = time.perf_counter()
#         output = loaded_model(data)
        
#         # Calculate and accumulate the loss
#         test_loss += F.cross_entropy(output, target, reduction='sum').item()
        
#         # Get the predicted class and count correct predictions
#         pred = output.argmax(dim=1, keepdim=True)
#         end_time = time.perf_counter()
#         elasped_time += end_time - start_time
#         correct += pred.eq(target.view_as(pred)).sum().item()

# # 6. Display the results
# test_loss /= len(subset_dataset)
# accuracy = 100. * correct / len(subset_dataset)

# print(f'\nEvaluation results on 100 images:')
# print(f'Average loss: {test_loss:.4f}')
# print(f'Accuracy: {correct}/{len(subset_dataset)} ({accuracy:.2f}%)')
# print(f'Average Computation Time: {elasped_time/num_tests} s')
