import argparse
import os
import numpy as np
import zipfile
import math
import torch
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

# DCGAN source: https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L18/04_02_dcgan-celeba.ipynb  

class DCGAN(torch.nn.Module):

    def __init__(self, latent_dim=100, 
                 num_feat_maps_gen=64, num_feat_maps_dis=64,
                 color_channels=3):
        super().__init__()
        
        
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_feat_maps_gen*8, 
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*8),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*8 x 4 x 4
            #
            nn.ConvTranspose2d(num_feat_maps_gen*8, num_feat_maps_gen*4, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*4),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*4 x 8 x 8
            #
            nn.ConvTranspose2d(num_feat_maps_gen*4, num_feat_maps_gen*2, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*2),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*2 x 16 x 16
            #
            nn.ConvTranspose2d(num_feat_maps_gen*2, num_feat_maps_gen, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen x 32 x 32
            #
            nn.ConvTranspose2d(num_feat_maps_gen, color_channels, 
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            #
            # size: color_channels x 64 x 64
            #  
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
                #
                # input size color_channels x image_height x image_width
                #
                nn.Conv2d(color_channels, num_feat_maps_dis,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                #
                # size: num_feat_maps_dis x 32 x 32
                #              
                nn.Conv2d(num_feat_maps_dis, num_feat_maps_dis*2,
                          kernel_size=4, stride=2, padding=1,
                          bias=False),        
                nn.BatchNorm2d(num_feat_maps_dis*2),
                nn.LeakyReLU(inplace=True),
                #
                # size: num_feat_maps_dis*2 x 16 x 16
                #   
                nn.Conv2d(num_feat_maps_dis*2, num_feat_maps_dis*4,
                          kernel_size=4, stride=2, padding=1,
                          bias=False),        
                nn.BatchNorm2d(num_feat_maps_dis*4),
                nn.LeakyReLU(inplace=True),
                #
                # size: num_feat_maps_dis*4 x 8 x 8
                #   
                nn.Conv2d(num_feat_maps_dis*4, num_feat_maps_dis*8,
                          kernel_size=4, stride=2, padding=1,
                          bias=False),        
                nn.BatchNorm2d(num_feat_maps_dis*8),
                nn.LeakyReLU(inplace=True),
                #
                # size: num_feat_maps_dis*8 x 4 x 4
                #   
                nn.Conv2d(num_feat_maps_dis*8, 1,
                          kernel_size=4, stride=1, padding=0),
                
                # size: 1 x 1 x 1
                nn.Flatten(),
                
            )

                
    def generator_forward(self, z):
        img = self.generator(z)
        return img
        
    def discriminator_forward(self, img):
        logits = model.discriminator(img)
        return logits


model = DCGAN()
model.to(device)

# With Adam, good results
optim_gen = torch.optim.Adam(model.generator.parameters(),
                             betas=(opt.b1, opt.b2),
                             lr=opt.lr)

optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                               betas=(opt.b1, opt.b2),
                               lr=opt.lr)

# With SGD and CosineAnnealingLR, at least, initially, it was bad: gradiant vanishing in Dis
"""
Mostly noisy pictures
torch.Size([128, 3, 64, 64])
Epoch: 001/030 | Batch 000/1272 | Gen/Dis Loss: 1.5112/0.7281
Epoch: 001/030 | Batch 025/1272 | Gen/Dis Loss: 46.0621/1.3114
Epoch: 001/030 | Batch 050/1272 | Gen/Dis Loss: 107.7352/0.7285
Epoch: 001/030 | Batch 075/1272 | Gen/Dis Loss: 67.2650/0.0029
Epoch: 001/030 | Batch 100/1272 | Gen/Dis Loss: 37.9752/0.0000
Epoch: 001/030 | Batch 125/1272 | Gen/Dis Loss: 54.6652/0.0593
Epoch: 001/030 | Batch 150/1272 | Gen/Dis Loss: 37.9497/0.0000
Epoch: 001/030 | Batch 175/1272 | Gen/Dis Loss: 25.4828/0.0000
Epoch: 001/030 | Batch 200/1272 | Gen/Dis Loss: 20.4135/0.0000
Epoch: 001/030 | Batch 225/1272 | Gen/Dis Loss: 17.4199/0.0029
Epoch: 001/030 | Batch 250/1272 | Gen/Dis Loss: 98.9853/0.0205
Epoch: 001/030 | Batch 275/1272 | Gen/Dis Loss: 38.9305/0.0991
Epoch: 001/030 | Batch 300/1272 | Gen/Dis Loss: 34.4951/0.0000
Epoch: 001/030 | Batch 325/1272 | Gen/Dis Loss: 31.7266/0.0001
Epoch: 001/030 | Batch 350/1272 | Gen/Dis Loss: 25.9539/0.0000
"""
# download the data directly
data_root = 'data/celeba'
dataset_folder = f'{data_root}'
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.CenterCrop((64, 64)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize tensors
])

train_loader, valid_loader, test_loader = get_dataloaders_celeba(
    batch_size=opt.batch_size,
    train_transforms=transform,
    test_transforms=transform,
    num_workers=4)

# check the data
print("length of train_loader", len(train_loader))
print("length of valid_loader", len(valid_loader))
print("length of test_loader", len(test_loader))

n = 0
dataiter = iter(train_loader)
images, labels = next(dataiter)
# check if it's a tensor or a list
print(type(images))  
print(images[:2])
print(images.shape)

os.makedirs("real_images", exist_ok=True)

# plot configuration
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64], padding=2, normalize=True), (1, 2, 0)))

# save the plot as a PNG file in the "real_images" folder, check if the input looks good
plt.savefig("real_images/training_images.png")

# show the plot
plt.show()

# Training function source: https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L18/helper_train.py 
log_dict = train_gan_v1(num_epochs=opt.n_epochs, model = model,
                        optimizer_gen = optim_gen,
                        optimizer_discr = optim_discr,
                        latent_dim=opt.latent_dim,
                        device=device, 
                        train_loader=train_loader,
                        logging_interval= 400,
                        save_model='gan_celeba_01.pt', 
                        save_images_dir="images", 
                        lr_scheduler= None)
                        
# os.makedirs("reports", exist_ok=True)
plot_multiple_training_losses(
    losses_list=(
        log_dict['train_discriminator_loss_per_batch'],
        log_dict['train_generator_loss_per_batch']
    ),
    num_epochs = opt.n_epochs,
    averaging_iterations=100,
    custom_labels_list=(' -- Discriminator', ' -- Generator'),
    save_dir="reports"
)

plot_accuracy_per_epoch(
                log_dict['train_discriminator_real_acc_per_epoch'], 
                log_dict['train_discriminator_fake_acc_per_epoch'], 
                num_epochs = opt.n_epochs, 
                save_dir = "reports")

plot_multiple_training_accuracies(
                log_dict['train_discriminator_real_acc_per_batch'], 
                log_dict['train_discriminator_fake_acc_per_batch'], 
                num_epochs = opt.n_epochs, 
                save_dir = "reports")

os.makedirs("images", exist_ok=True)

# save images generated at epoch intervals
for i in range(0, NUM_EPOCHS, 5):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(f'Generated images at epoch {i}')
    plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][i], (1, 2, 0)))
    plt.savefig(f"images/generated_images_epoch_{i}.png")
    plt.close()  # Close the figure to free up memory

# Save images generated after the last epoch
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title(f'Generated images after last epoch')
plt.imshow(np.transpose(log_dict['images_from_noise_per_epoch'][-1], (1, 2, 0)))
plt.savefig("images/generated_images_last_epoch.png")
plt.close()  # Close the figure to free up memory
