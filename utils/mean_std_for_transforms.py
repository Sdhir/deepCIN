import torch
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import os
import sys
import numpy as np

image_DIR = r'/usr/local/home/ssbw5/classification/data/x_validation/resz_64_704'

data_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor()
])

image_datasets = datasets.ImageFolder(os.path.join(image_DIR),transform=data_transform)

dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=11667, 
                    shuffle=False, sampler = None, num_workers=2)

pop_mean = []
pop_std0 = []
pop_std1 = []
for i, data in enumerate(dataloader):
    # shape (batch_size, 3, height, width)
    numpy_image = data[0].numpy()
    print(numpy_image.shape)
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std0 = np.std(numpy_image, axis=(0,2,3))
    batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
pop_std1 = np.array(pop_std1).mean(axis=0)

print(pop_mean, pop_std0, pop_std1)