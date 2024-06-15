import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
classes = ['cat', 'dog']


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

PATH_train = r'C:\python自學\深度學習 資料\Cat_Dog_data\train'
PATH_val = r"C:\python自學\深度學習 資料\Cat_Dog_data\Cat_Dog_data"
PATH_test = r"C:\python自學\深度學習 資料\Cat_Dog_data\test"
TRAIN = Path(PATH_train)
VALID = Path(PATH_val)
TEST = Path(PATH_test)
print(TRAIN)
print(VALID)
print(TEST)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# learning rate
LR = 0.01

# Convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID, transform=valid_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)

print(train_data.class_to_idx)
print(valid_data.class_to_idx)

# Prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

# Define a function to denormalize the images
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def denormalize(image):
    image = transforms.Normalize(-mean / std, 1 / std)(image)  # Denormalize
    image = image.permute(1, 2, 0)  # Change from 3x224x224 to 224x224x3
    image = torch.clamp(image, 0, 1)
    return image

# Helper function to un-normalize and display an image
def imshow(img):
    img = denormalize(img)
    plt.imshow(img)

# Obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 8))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx].item()])
plt.show()