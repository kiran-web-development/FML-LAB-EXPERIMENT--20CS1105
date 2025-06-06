# Aim :- Write a program for implementing the classification using Multilayer perceptron.



#Program:-

import numpy as np
import pandas as pd
import os
import copy
import time
import torch
import torch nn as nn import cv2
import matplotlib.pyplot as plt import copy
import time
import albumentations as A
import torch_optimizer as optim from res_mlp_pytorch
import ResMLP
from PIL import Image
from albumentations.pytorch
import ToTensorV2
 from torch.utils.data
import Dataset, DataLoader
class FoodDataset(Dataset):
def	init (self, data_type=None, transforms=None): self.path = 'Classification using Multilayer perceptron\food5k' + data_type + '/' self.images_name = os.listdir(self.path) self.transforms = transforms
   def	len (self):
return len(self.images_name)
def getitem (self, idx):
data = self.images_name[idx]
label = data.split('_')[0]
label = int(label)
 label = torch.tensor(label)
image = cv2.imread(self.path + data)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if self.transforms: aug =self.transforms(imag e=image)image = aug['image']
return (image, label)
                train_data = FoodDataset('training', A.Compose([ A.RandomResized Crop(256, 256),
            A.HorizontalFlip(), A.Normalize(), ToTensorV2()]))
                val_data = FoodDataset('validation', A.Compose([ A.Resize(384, 384),
                                                        A.CenterCrop(256, 256), A.Normalize(), ToTensorV2()]))
    dataloaders = {
'train': DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4), 'val': DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4), 'test': DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)
}

dataset_sizes = {'train': len(train_data), 'val': len(val_data),'test': len(test_data)
def train_model(model, criterion, optimizer, epochs=1): 
since = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 0.0
  best_acc = 0
for ep in range(epochs): print(f"Epoch {ep}/{epochs-1}")print("-"*10) 
for phase in ['train', 'val']:
if phase == 'train':model.train()
else:
model.eval()
running_loss = 0.0
running_corrects = 0

for images, labels in dataloaders[phase]:
images = images.to(device)
labels = labels.to(device)
optimizer.zero_grad() with torch.set_grad_enabled(phase == 'train'):
outputs = model(images)
_, preds = torch.max(outpu ts, 1)
loss = criterion(output s, labels)
if phase == 'train':loss.backward() optimizer.step() 
running_loss += loss.item() *images.size(0)
 running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects.double() / dataset_sizes[phase] print(f"{phase} Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f}")
if phase == 'val': 
if ep == 0:

best_loss = epoch_loss
best_model_wts=copy.deepcopy(model.state_dict())
else:
if epoch_loss < best_loss:
best_loss = epoch_loss
best_acc = epoch_acc 
best_model_wts = copy.deepcopy(model.state_dict())

print()

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60}m{time_elapsed % 60}s')
print(f'Best val loss:{best_loss:.4f}')
 print(f'Best acc:{best_acc}')
model.load_state_dict(b est_model_wts)
return model
    # Train The Model
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = ResMLP(image_size=256, patch_size=16, dim=512, depth=12,num_classes=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Lamb(model.parameters(), lr=0.005, weight_decay=0.2)
best_model = train_model(model, criterion, optimizer, epochs=20)





"""
food5k/
├── training/
│   ├── 0_image1.jpg
│   ├── 0_image2.png
│   ├── 1_image3.jpg
│   └── ...
└── validation/
    ├── 0_validation_image1.jpg
    ├── 1_validation_image2.png
    └── ...
    """



#Note:-That this program was not executing successfully due to some errors in the code.remaing all the programs were executed successfully.


#Result:-The above program is not executed successfully due to some errors in the code.