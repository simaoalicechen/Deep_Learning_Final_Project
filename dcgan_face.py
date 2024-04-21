import argparse
import os
import numpy as np
import zipfile
import math
import torch
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils import load_dataset
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default = 200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# DCGAN modification source: https://github.com/joeylitalien/celeba-gan-pytorch/blob/master/src/dcgan.py#L213
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Project and reshape:
        self.linear = nn.Sequential(
          nn.Linear(latent_dim, 512*4*4, bias = False),
          nn.BatchNorm1d(512*4*4),
          nn.ReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
          nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding=1, bias=False),
          nn.BatchNorm2d(256), 
          nn.ReLU(inplace = True),
          nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
          nn.Tanh())

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1))

    def forward(self, x):
        return self.features(x).view(-1)
            # x = self.features(x)  
            # x = x.view(x.size(0), -1)  
            # x = torch.sigmoid(x) 
            # return x

    def clip(self, c=0.05):
        """Weight clipping in (-c, c)"""

        for p in self.parameters():
            p.data.clamp_(-c, c)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Loss function
# adversarial_loss = torch.nn.BCELoss()
# adversarial_loss = F.binary_cross_entropy_with_logits()
generator = Generator(opt.latent_dim)
discriminator = Discriminator()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

generator = generator.to(device)
discriminator = discriminator.to(device)

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# CelebA
# The custom dataloader codes are from: https://stackoverflow.com/questions/65528568/how-do-i-load-the-celeba-dataset-on-google-colab-using-torch-vision-without-ru

data_root = 'data/celebA'
dataset_folder = f'{data_root}'
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.CenterCrop((64, 64)),  # Crop to remove unwanted borders
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize tensors
])
celeba_dataset = ImageFolder(root=data_root,
            transform=transform)

dataloader = DataLoader(celeba_dataset, batch_size=32, shuffle=True, num_workers=4)

print("length of celeba_dataloader", len(dataloader))
n = 0
dataiter = iter(dataloader)
images, labels = next(dataiter)
print(type(images))  # Check if it's a tensor or a list
print(images[:2])
print(images.shape)  # Should only contain images


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
criterion = nn.BCELoss()

os.makedirs("real_images", exist_ok=True)
for epoch in range(opt.n_epochs):
    for i, (images, attributes) in enumerate(dataloader):
        # Save the first 10 batches of images
        # if i < 5:
        #     # Save the images
        #     torchvision.utils.save_image(images, f"real_images/batch_{i}.png", nrow=8, normalize=True)
        #     print(f"Batch {i} of real images saved.")
        # else:
        #     break  # Stop after saving 10 batches
            # print(images.shape)
            # print(attributes.shape)
        real_images = images.to(device)
        batch_size = real_images.size(0)
        # Create labels
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss for real images
        outputs = discriminator(real_images)
        d_loss_real = F.binary_cross_entropy_with_logits(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(batch_size, 100).to(device)  # 100 is the size of the latent vector
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = F.binary_cross_entropy_with_logits(outputs, fake_labels)
        fake_score = outputs

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # =============================================
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        # =============================================
        optimizer_G.zero_grad()
        
        # Loss for fake images
        outputs = discriminator(fake_images)
        g_loss = F.binary_cross_entropy_with_logits(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{opt.n_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
        
        if i % opt.sample_interval == 0:
            # with torch.no_grad():
            # sample_images = generator(fixed_noise).detach().cpu()
            fake_images = fake_images.detach().cpu()
            img_grid = torchvision.utils.make_grid(fake_images, nrow=5, normalize=True)
            torchvision.utils.save_image(img_grid, f"images/epoch_{epoch}_batch_{i}.png")
            plt.figure(figsize=(10,10))
            plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
            plt.axis('off')
            plt.show()