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
parser.add_argument("--n_epochs", type=int, default = 3000, help="number of epochs of training")
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

# This architecture codes are from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

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


report_dir = "report_WGAN64_CelebA"
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

os.makedirs("report_WGAN64_CelebA", exist_ok=True)

# define parameters for metrics and graphs
all_d_losses, all_g_losses = [], []
all_real_scores, all_fake_scores = [], []
all_real_accs, all_fake_accs = [], []

# Path to save the models
save_path = 'saves_WGAN64/'
os.makedirs(save_path, exist_ok=True)

# TODO
# each time, check what the latest saved epoch was and get it from the checkpoint, and then 
# re-start training from that epoch
checkpoint_path = 'saves_WGAN64/checkpoint_epoch_104.pth'  

def load_checkpoint(filepath, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint = torch.load(filepath)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch']
    lossG = checkpoint['lossG']
    lossD = checkpoint['lossD']
    real_score = checkpoint['real_score']
    fake_score = checkpoint['fake_score']
    real_acc = checkpoint['real_acc']
    fake_acc = checkpoint['fake_acc']
    return start_epoch, lossG, lossD, real_score, fake_score, real_acc, fake_acc

start_epoch, lossG, lossD, real_score, fake_score, real_acc, fake_acc = load_checkpoint(
        checkpoint_path, generator, discriminator, optimizer_G, optimizer_D
    )
print("start_epoch is: ", start_epoch)

threshold = 0.5
# training loop
for epoch in range(start_epoch, opt.n_epochs):
    d_losses, g_losses = [], []
    real_scores, fake_scores = [], []
    real_accs, fake_accs = [], []
    num_batches = 0
    start_time = time.time()
    for i, (real_images, _) in enumerate(train_loader):
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

        real_score = torch.sigmoid(real_output).mean().item()
        fake_score = torch.sigmoid(fake_output).mean().item()
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        real_acc = (torch.sigmoid(real_output).round().cpu().detach().numpy() == 1).mean()
        fake_acc = (torch.sigmoid(fake_output).round().cpu().detach().numpy() == 0).mean()
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)

        # calculate loss for discriminator
        d_loss = -(torch.mean(real_output) - torch.mean(fake_output))
        d_losses.append(d_loss.item())

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

        g_losses.append(g_loss.item())
        # Print batch loss, discriminator scores, and accuracy
        print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {i+1}/{len(train_loader)}] "
              f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] "
              f"[D's scores on real images: {real_score:.6f}] [fake score: {fake_score:.6f}] "
              f"[D's accuracies on real images: {real_acc:.4%}] [fake images: {fake_acc:.4%}]")

    # calculate average loss and scores for the current epoch
    epoch_d_loss = sum(d_losses) / len(d_losses)
    epoch_g_loss = sum(g_losses) / len(g_losses)
    epoch_real_score = sum(real_scores) / len(real_scores)
    epoch_fake_score = sum(fake_scores) / len(fake_scores)
    epoch_real_acc = sum(real_accs)/len(real_accs)
    epoch_fake_acc = sum(fake_accs)/len(fake_accs)

    # store epoch metrics data
    all_d_losses.append(epoch_d_loss)
    all_g_losses.append(epoch_g_loss)
    all_real_scores.append(epoch_real_score)
    all_fake_scores.append(epoch_fake_score)
    all_real_accs.append(epoch_real_acc)
    all_fake_accs.append(epoch_fake_acc)

    torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'lossG': g_loss.item(),
            'lossD': d_loss.item(),
            'real_score': epoch_real_score,       
            'fake_score': epoch_fake_score,     
            'real_acc': epoch_real_acc,            
            'fake_acc': epoch_fake_acc            
    }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))


    # training stats for the epoch
    print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {i+1}/{len(train_loader)}] "
            f"[D loss: {epoch_d_loss:.6f}] [G loss: {epoch_g_loss:.6f}] "
            f"[D's epoch mean scores on real images: {epoch_real_score:.6f}] [fake images: {epoch_fake_score:.6f}] "
            f"[D's epoch mean accuracies on real images: {epoch_real_acc:.4%}] [fake images: {epoch_fake_acc:.4%}]")


    # end timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60.0
    print(f"Epoch {epoch+1} took {elapsed_minutes:.2f} minutes.")

    # generate and save fake images
    with torch.no_grad():
        noise = torch.randn(64, opt.latent_dim).to(device)
        fake = generator(noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.savefig(os.path.join("report_WGAN64_CelebA/images", f"epoch_{epoch+1}_generated_images.png"))
        plt.close()

    # define the directory path
    directoryL = "report_WGAN64_CelebA/losses"
    directoryS = "report_WGAN64_CelebA/scores"
    directoryA = "report_WGAN64_CelebA/accuracies"

    # create the directory if it doesn't exist
    if not os.path.exists(directoryL):
        os.makedirs(directoryL)
    if not os.path.exists(directoryA):
        os.makedirs(directoryA)
    if not os.path.exists(directoryS):
        os.makedirs(directoryS)

    if (epoch+1) in [1, 10, 30, 50]:
        # losses
        plt.figure(figsize=(10, 5))
        plt.plot(all_d_losses, label='Discriminator Loss')
        plt.plot(all_g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator and Generator Losses')
        plt.legend()
        plt.savefig(os.path.join("report_WGAN64_CelebA/losses", f"losses_graph_epoch_{epoch+1}.png"))
        plt.close()

        # accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(all_real_accs, label='Real Image Accuracy')
        plt.plot(all_fake_accs, label='Fake Image Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.title("Discriminator's Accuracies of Real and Fake Images")
        plt.legend()
        plt.savefig(os.path.join("report_WGAN64_CelebA/accuracies", f"accuracy_graph_epoch_{epoch+1}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(all_real_scores, label='real image Scores')
        plt.plot(all_fake_scores, label='fake image Scores')
        plt.xlabel('Epochs')
        plt.ylabel('Scores')
        plt.title('Discriminator Scores on real and fake images')
        plt.legend()
        plt.savefig(os.path.join("report_WGAN64_CelebA/scores", f"scores_graph_epoch_{epoch + 1}.png"))
        plt.close()
