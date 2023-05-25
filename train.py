# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import itertools

from model import Generator, Discriminator  # assuming you have these modules defined in model.py

# Hyperparameters
lr_D = 0.0001
lr_G = 0.0003
batch_size = 32
image_size = 256
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Data loader for domain A and B
data_loader_A = DataLoader(datasets.ImageFolder('thermal2rgb/trainA', transform),
                         batch_size=batch_size,
                         shuffle=True)

data_loader_B = DataLoader(datasets.ImageFolder('thermal2rgb/trainB', transform),
                         batch_size=batch_size,
                         shuffle=True)

# Initialize CycleGAN generators and discriminators
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Optimizers
opt_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr_G)
opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr_D)

criterion_GAN = torch.nn.MSELoss().to(device)  # for adversarial loss
criterion_cycle = torch.nn.L1Loss().to(device)  # for cycle consistency loss

# Training
for epoch in range(num_epochs):
    for i, ((real_images_A, _), (real_images_B, _)) in enumerate(zip(data_loader_A, data_loader_B)):
        if real_images_A is None or real_images_B is None:
            continue  # Skip the rest of this iteration

        real_images_A = real_images_A.to(device)
        real_images_B = real_images_B.to(device)

        # Forward pass
        fake_images_B = G_A2B(real_images_A)
        cycle_images_A = G_B2A(fake_images_B)

        fake_images_A = G_B2A(real_images_B)
        cycle_images_B = G_A2B(fake_images_A)

        # GAN loss
        fake_outputs_B = D_B(fake_images_B)
        loss_GAN_A2B = criterion_GAN(fake_outputs_B, torch.ones_like(fake_outputs_B))

        fake_outputs_A = D_A(fake_images_A)
        loss_GAN_B2A = criterion_GAN(fake_outputs_A, torch.ones_like(fake_outputs_A))

        # Cycle loss
        loss_cycle_A2B = criterion_cycle(cycle_images_A, real_images_A)
        loss_cycle_B2A = criterion_cycle(cycle_images_B, real_images_B)

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_cycle_A2B + loss_GAN_B2A + loss_cycle_B2A

        # Backward pass and optimization for generators
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # Forward pass for discriminator
        real_outputs_B = D_B(real_images_B)
        loss_real_B = criterion_GAN(real_outputs_B, torch.ones_like(real_outputs_B))
        fake_outputs_B = D_B(fake_images_B.detach())
        loss_fake_B = criterion_GAN(fake_outputs_B, torch.zeros_like(fake_outputs_B))

        real_outputs_A = D_A(real_images_A)
        loss_real_A = criterion_GAN(real_outputs_A, torch.ones_like(real_outputs_A))
        fake_outputs_A = D_A(fake_images_A.detach())
        loss_fake_A = criterion_GAN(fake_outputs_A, torch.zeros_like(fake_outputs_A))

        # Total discriminator loss
        loss_D_A = (loss_real_A + loss_fake_A) / 2
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D = loss_D_A + loss_D_B

        # Backward pass and optimization for discriminators
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        # Save loss values for this batch
        loss_G_values.append(loss_G.item())
        loss_D_values.append(loss_D.item())
        loss_D_A_values.append(loss_D_A.item())
        loss_D_B_values.append(loss_D_B.item())

    # Save some generated images and the model
    if (epoch + 1) % 10 == 0:
        save_image(fake_images_B, f'images/fake_images_A2B-{epoch + 1}.png', normalize=True)
        save_image(fake_images_A, f'images/fake_images_B2A-{epoch + 1}.png', normalize=True)
        torch.save(G_A2B.state_dict(), f'models/G_A2B-{epoch + 1}.ckpt')
        torch.save(G_B2A.state_dict(), f'models/G_B2A-{epoch + 1}.ckpt')
        torch.save(D_A.state_dict(), f'models/D_A-{epoch + 1}.ckpt')
        torch.save(D_B.state_dict(), f'models/D_B-{epoch + 1}.ckpt')

print('Training completed.')

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(loss_G_values, label='Generator loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(loss_D_values, label='Discriminator total loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(loss_D_A_values, label='Discriminator A loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(loss_D_B_values, label='Discriminator B loss')
plt.legend()

plt.show()
