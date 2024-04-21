import argparse
import os
import numpy as np
import zipfile 
import gdown
import torch
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import math
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
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
parser.add_argument("--sample_interval", type=int, default=25, help="interval betwen image samples")
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
            x = self.features(x)  
            x = x.view(x.size(0), -1)  
            x = torch.sigmoid(x) 
            return x

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


# Root directory for the dataset
data_root = 'data/celebA'
dataset_folder = f'{data_root}'
zip_data_folder = f'{data_root}'

## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

## Load the dataset 
# Path to directory with all the images
img_folder = f'{dataset_folder}/img_align_celeba'
# Spatial size of training images, images are resized to this size.
image_size = 64
# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)

## Create a dataloader 
# Batch size during training
batch_size = 128
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if device.type == 'cuda' else False

celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)

print(len(celeba_dataloader))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# Just show one batch of real input data
def imshow(img, filename='output_image.png'):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8,8))  # Set the figure size to be larger
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Reorder dimensions for matplotlib
    plt.axis('off')  # Turn off the axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # Save the figure as a PNG file
    plt.show()

# Get some random training images
dataiter = iter(celeba_dataloader)
images = next(dataiter)  # Use next() function to get the next batch

# Display and save the grid of images
imshow(torchvision.utils.make_grid(images), 'sample_grid.png')

print("length of celeba_dataloader", len(celeba_dataloader))
n = 0
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(celeba_dataloader):
        batch = batch.to(device)
        print(i)
        print(len(batch))
        print(type(batch))
        print(batch.shape)
        imgs = batch


        # Adversarial ground truths
        # valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).detach()
        # fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).detach()
        z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)  # Directly create on GPU
        valid = torch.full((imgs.size(0), 1), 1.0, device=device)  # Also directly create on GPU
        fake = torch.full((imgs.size(0), 1), 0.0, device=device)

        # Configure input
        real_imgs = imgs.type(Tensor)

        print("Shape of real images:", real_imgs.shape)
        print("Shape of validity labels:", valid.shape)

        real_loss = F.binary_cross_entropy_with_logits(discriminator(real_imgs), valid)
      

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = F.binary_cross_entropy_with_logits(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = F.binary_cross_entropy_with_logits(discriminator(real_imgs), valid)
        fake_loss = F.binary_cross_entropy_with_logits(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(celeba_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(celeba_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print(f"pictures {n}")
            n += 1
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)