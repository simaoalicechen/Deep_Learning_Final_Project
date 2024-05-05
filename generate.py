import argparse
import os
import numpy as np
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
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# Device
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')
cuda = True if torch.cuda.is_available() else False
RANDOM_SEED = 42
set_all_seeds(RANDOM_SEED) 

ssl._create_default_https_context = ssl._create_unverified_context
use_gpu = True if torch.cuda.is_available() else False

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)

# optimizer
optim_gen = model.getOptimizerG()
optim_discr = model.getOptimizerD()

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


report_dir = "report2"
os.makedirs(report_dir, exist_ok=True)

# initialize as tensor, to avoid no .item() error
loss_real = torch.tensor(0.0)
loss_fake = torch.tensor(0.0)
loss_G = torch.tensor(0.0)

num_epochs = 50
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        # train Discriminator
        optim_discr.zero_grad()
        real_images, _ = data
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        label_real = torch.ones(batch_size, 1).to(device)
        label_fake = torch.zeros(batch_size, 1).to(device)

        # train with real images
        output_real = model.netD(real_images)
        print("output_real.shape: ", output_real.shape)
        print("label_real.shape: ", label_real.shape)
        break
        label_real = torch.ones_like(output_real)
        loss_real = criterion(output_real, label_real)
        loss_real.backward()

        # train with fake images
        noise = torch.randn(batch_size, 120, 1, 1).to(device)
        fake_images = model.netG(noise)

        output_fake = model.netD(fake_images.detach())
        label_fake = torch.zeros_like(output_fake)
        loss_fake = criterion(output_fake, label_fake)
        loss_fake.backward()

        optim_discr.step()

        # train Generator
        optim_gen.zero_grad()
        output = model.netD(fake_images)
        loss_G = criterion(output, label_real)  # Generator tries to fool the discriminator
        loss_G.backward()
        optim_gen.step()

    print('[%d/%d] Loss_D_Real: %.4f Loss_D_Fake: %.4f Loss_G: %.4f' %
          (epoch + 1, num_epochs, loss_real.item(), loss_fake.item(), loss_G.item()))

    
    with torch.no_grad():
        noise = torch.randn(batch_size, 120, 1, 1)
        if use_gpu:
            noise = noise.cuda()
        fake = model.netG(noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.savefig(os.path.join(report_dir, f"epoch_{epoch}_generated_images.png"))
        plt.close()










