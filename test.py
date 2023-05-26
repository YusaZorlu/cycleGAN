# test.py
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import Generator  # assuming you have these modules defined in model.py
import os

# Hyperparameters
batch_size = 32
image_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('test_results'):
    os.makedirs('test_results')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Data loader for domain A and B
data_loader_A = DataLoader(datasets.ImageFolder('test_a', transform),
                           batch_size=batch_size,
                           shuffle=True)

data_loader_B = DataLoader(datasets.ImageFolder('test_b', transform),
                           batch_size=batch_size,
                           shuffle=True)

# Initialize CycleGAN generators
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)

# Load the models
G_A2B.load_state_dict(torch.load('models/G_A2B-10.ckpt'))  # replace with the path to your model
G_B2A.load_state_dict(torch.load('models/G_B2A-10.ckpt'))  # replace with the path to your model

# Generate and save images
for i, ((real_images_A, _), (real_images_B, _)) in enumerate(zip(data_loader_A, data_loader_B)):
    if real_images_A is None or real_images_B is None:
        continue  # Skip the rest of this iteration

    real_images_A = real_images_A.to(device)
    real_images_B = real_images_B.to(device)

    with torch.no_grad():  # No need to calculate gradients during testing
        fake_images_B = G_A2B(real_images_A)
        fake_images_A = G_B2A(real_images_B)

    save_image(fake_images_B, f'test_results/fake_images_A2B-{i}.png', normalize=True)
    save_image(fake_images_A, f'test_results/fake_images_B2A-{i}.png', normalize=True)

print('Testing completed.')
