import os, certifi

from PIL import Image

import os, torch, glob, pandas as pd 

from torchvision import datasets, transforms

from torchvision.models import resnet50, ResNet50_Weights

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit

import torch.optim as optim
import torch, torch.nn as nn

class CSVDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['path']).convert('RGB')
        y = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, y


file_path_train = '/Users/khizernaeem/Downloads/state-farm-distracted-driver-detection (1)/imgs/train'

IMG = 224

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(IMG, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG,IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225]),
                                    
])


csv_path = '/Users/khizernaeem/Downloads/state-farm-distracted-driver-detection (1)/driver_imgs_list.csv'
df = pd.read_csv(csv_path)

# Build absolute file paths and numeric labels
df['path'] = df.apply(lambda r: os.path.join(file_path_train, r['classname'], r['img']), axis=1)
class_names = sorted(df['classname'].unique())                 # e.g. ['c0',...,'c9']
class_to_idx = {c:i for i,c in enumerate(class_names)}         # {'c0':0,...}
df['label'] = df['classname'].map(class_to_idx)

print(df.head())
print("Drivers:", df['subject'].nunique(), "Classes:", class_names)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=35)
groups = df['subject'].values
train_idx, val_idx = next(gss.split(df, groups=groups))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)

print("Train drivers:", df_train['subject'].nunique(),
      "Val drivers:", df_val['subject'].nunique())
print("Val driver IDs:", sorted(df_val['subject'].unique()))



train_ds = CSVDataset(df_train, transform=train_transform)
val_ds   = CSVDataset(df_val,   transform=val_transform)

# MPS: keep workers small (0–2); CUDA: you can use 2–4 and pin_memory=True
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)


device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = resnet50(weights= ResNet50_Weights.IMAGENET1K_V1)



model.fc = nn.Linear(model.fc.in_features, out_features= 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
                      

epochs = 5

for epoch in range(epochs): 
  model.train()
  running_loss = 0
  running_correct = 0
  seen = 0                      
  for inputs, label in train_loader:

      inputs, label = inputs.to(device), label.to(device)

      optimizer.zero_grad(set_to_none=True)

      outputs = model(inputs)

      loss = criterion(outputs, label) 

      loss.backward()

      optimizer.step() # updates the weights with the correct parameters 

      running_loss   += loss.item() * label.size(0)
      preds           = outputs.argmax(1)
      running_correct += (preds == label).sum().item()
      seen            += label.size(0)

  epoch_train_loss = running_loss / seen
  epoch_train_acc  = running_correct / seen
  train_losses.append(epoch_train_loss)
  train_accs.append(epoch_train_acc)
  model.eval()
  v_loss = 0.0
  v_correct = 0
  v_seen = 0

  with torch.no_grad():

    for val_input, val_label in val_loader: 

      val_input, val_label = val_input.to(device), val_label.to(device)
      
      val_outputs = model(val_input)

      loss = criterion(val_outputs, val_label)

      v_loss += loss.item() * val_label.size(0)
      v_preds = val_outputs.argmax(1)
      v_correct += ( v_preds == val_label).sum().item()
      v_seen += val_label.size(0)

    epoch_val_loss = v_loss / v_seen
    epoch_val_acc  = v_correct / v_seen
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
          f"Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%")
      



    


                
