import argparse
import os
import numpy as np
import time
import zipfile
import math
import torch
import ssl
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image
import gdown
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
from dcgan_model import DCGAN
from utils import load_dataset, get_dataloaders_celeba, set_all_seeds, set_deterministic
from helper_train import train_gan_v1
from helper_plotting import plot_multiple_training_losses, plot_multiple_training_accuracies, plot_accuracy_per_epoch, plot_multiple_training_accuracies
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default = 50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--lr", type=float, default=0.0002, help="SGD: learning rate")
# parser.add_argument("--lr", type=float, default=0.01, help="SGD: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# device
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')
cuda = True if torch.cuda.is_available() else False
RANDOM_SEED = 42
set_all_seeds(RANDOM_SEED) 

img_shape = (opt.channels, opt.img_size, opt.img_size)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        # img = img.view(img.shape[0], 3, 64, 64)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# optimizer
optim_gen = Generator().to(device)
optim_discr = Discriminator().to(device)

# loss function 
criterion = nn.BCELoss()
# criterion = nn.MSELoss()

# download the data directly
data_root = 'data/celeba'
dataset_folder = f'{data_root}'
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.CenterCrop((64, 64)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# dataloader = DataLoader(celeba_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
train_loader, valid_loader, test_loader = get_dataloaders_celeba(
    batch_size=opt.batch_size,
    train_transforms=transform,
    test_transforms=transform,
    num_workers=4)


report_dir = "reportWGan"
os.makedirs(report_dir, exist_ok=True)

# initialize as tensor, to avoid no .item() error
loss_real = torch.tensor(0.0)
loss_fake = torch.tensor(0.0)
loss_G = torch.tensor(0.0)

num_epochs = 50

# generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# loss function
criterion = nn.BCEWithLogitsLoss()

os.makedirs("reportWGAN", exist_ok=True)


# # training loop
# training loop
for epoch in range(opt.n_epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0
    num_batches = 0
    start_time = time.time()
    for i, (real_images, _) in enumerate(train_loader):
        d_loss = 0.0
        g_loss = 0.0
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # train discriminator
        optimizer_D.zero_grad()

        # generate fake images
        z = torch.randn(batch_size, opt.latent_dim, device=device)
        fake_images = generator(z).detach()

        # calculate discriminator outputs for real and fake images
        real_output = discriminator(real_images).view(-1)
        fake_output = discriminator(fake_images).view(-1)

        # calculate loss for discriminator
        d_loss = -(torch.mean(real_output) - torch.mean(fake_output))

        # update discriminator weights
        d_loss.backward()
        optimizer_D.step()

        # clip discriminator weights
        for param in discriminator.parameters():
            param.data.clamp_(-0.01, 0.01)

        # train generator every n_critic iterations
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()

            # generate fake images
            z = torch.randn(batch_size, opt.latent_dim, device=device)
            fake_images = generator(z)

            # calculate discriminator output for fake images
            fake_output = discriminator(fake_images).view(-1)

            # calculate loss for generator
            g_loss = -torch.mean(fake_output)

            # update generator weights
            g_loss.backward()
            optimizer_G.step()

        # Print batch loss metrics
        print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {d_loss:.6f}] [G loss: {g_loss:.6f}]")

        # Accumulate loss values
        total_d_loss += d_loss
        total_g_loss += g_loss
        num_batches += 1

    # Calculate average loss for the epoch
    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches

    # Output training stats for the epoch
    print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Avg D loss: {avg_d_loss:.6f}] [Avg G loss: {avg_g_loss:.6f}]")

    # Reset total loss for the next epoch
    total_d_loss = 0.0
    total_g_loss = 0.0

    # End timer for epoch
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60.0
    print(f"Epoch {epoch+1} took {elapsed_minutes:.2f} minutes.")

    # Generate and save fake images
    with torch.no_grad():
        noise = torch.randn(batch_size, opt.latent_dim).to(device)
        fake = generator(noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.savefig(os.path.join("reportWGAN", f"epoch_{epoch+1}_generated_images.png"))
        plt.close()




# for epoch in range(opt.n_epochs):
#     total_d_loss = 0.0
#     total_g_loss = 0.0
#     num_batches = 0
#     start_time = time.time()
#     for i, (real_images, _) in enumerate(train_loader):
#         d_loss = 0.0
#         g_loss = 0.0
#         real_images = real_images.to(device)
#         batch_size = real_images.size(0)

#         # train discriminator
#         optimizer_D.zero_grad()

#         # generate fake images
#         z = torch.randn(batch_size, opt.latent_dim, device=device)
#         fake_images = generator(z).detach()

#         # calculate discriminator outputs for real and fake images
#         real_output = discriminator(real_images).view(-1)
#         fake_output = discriminator(fake_images).view(-1)

#         # calculate loss for discriminator
#         d_loss = -(torch.mean(real_output) - torch.mean(fake_output))

#         # update discriminator weights
#         d_loss.backward()
#         optimizer_D.step()

#         # clip discriminator weights
#         for param in discriminator.parameters():
#             param.data.clamp_(-0.01, 0.01)

#         # train generator every n_critic iterations
#         if i % opt.n_critic == 0:
#             optimizer_G.zero_grad()

#             # generate fake images
#             z = torch.randn(batch_size, opt.latent_dim, device=device)
#             fake_images = generator(z).detach()

#             # calculate discriminator output for fake images
#             fake_output = discriminator(fake_images).view(-1)

#             # calculate loss for generator
#             g_loss = -torch.mean(fake_output)

#             # update generator weights
#             g_loss.backward()
#             optimizer_G.step()

#         # # Output training stats
#         # if i % opt.sample_interval == 0:
#         #     print(
#         #         f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
#         #     )
#         # Accumulate loss values
#     total_d_loss += d_loss
#     total_g_loss += g_loss
#     num_batches += 1
#     # output training stats for each epoch
#     avg_d_loss = total_d_loss / len(train_loader)
#     avg_g_loss = total_g_loss / len(train_loader)
#     print(f"[Epoch {epoch+1}/{opt.n_epochs}] [D loss: {avg_d_loss:.6f}] [G loss: {avg_g_loss:.6f}]")

#     # reset total loss for the next epoch
#     total_d_loss = 0.0
#     total_g_loss = 0.0

#     # End timer for epoch
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     elapsed_minutes = elapsed_time / 60.0
#     print(f"Epoch {epoch+1} took {elapsed_minutes:.2f} minutes.")

#     with torch.no_grad():
#         noise = torch.randn(batch_size, opt.latent_dim).to(device)
#         fake = generator(noise).detach().cpu()
#         img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
#         plt.imshow(np.transpose(img_grid, (1, 2, 0)))
#         plt.savefig(os.path.join("reportWGAN", f"epoch_{epoch+1}_generated_images.png"))
#         plt.close()
