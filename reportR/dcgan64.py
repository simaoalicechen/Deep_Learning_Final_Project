import argparse
import os
import numpy as np
import zipfile
import mathdcgan_revisited.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image
import gdown
import time
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
parser.add_argument("--n_epochs", type=int, default = 1000, help="number of epochs of training")
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

print(hasattr(model, 'generator'))  # Should print True if your model has a 'generator'
print(hasattr(model, 'discriminator'))

# dummy_images = torch.randn(128, 3, 128, 128, device=device)  
# output = model.discriminator_forward(dummy_images)
# print("Final output from discriminator:", output)
# With Adam, good results
optim_gen = torch.optim.Adam(model.generator.parameters(),
                             betas=(opt.b1, opt.b2),
                             lr=opt.lr)

optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                               betas=(opt.b1, opt.b2),
                               lr=opt.lr)

# With SGD and CosineAnnealingLR, at least, initially, it was bad: gradiant vanishing in Dis
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
# if loss_fn is None:
# loss_fn = F.binary_cross_entropy_with_logits

fixed_noise = torch.randn(64, opt.latent_dim, 1, 1, device=device)

start_time = time.time()
logging_interval = 200
num_epochs = opt.n_epochs

# if save_model is not None:
#     torch.save(model.state_dict(), save_model)
#     os.makedirs("reportCW/losses", exist_ok=True)
os.makedirs("reportR2/images", exist_ok=True)
# Path to save the models
save_path = 'savesR/'
os.makedirs(save_path, exist_ok=True)

# define parameters for metrics and graphs
all_d_losses, all_g_losses = [], []
all_real_scores, all_fake_scores = [], []
all_real_accs, all_fake_accs = [], []

for epoch in range(0, num_epochs+1):
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': [], 
                'train_discriminator_real_acc_per_epoch':[],
                'train_discriminator_fake_acc_per_epoch': []}

    epoch_real_acc = 0.0 
    epoch_fake_acc = 0.0  
    d_losses, g_losses = [], []
    real_scores, fake_scores = [], []
    real_accs, fake_accs = [], []
    num_batches = 0
    model.train()
    for batch_idx, (features, _) in enumerate(train_loader):
        num_batches += 1
        batch_size = features.size(0)

        # real images
        real_images = features.to(device)
        print(real_images.shape)
        real_labels = torch.ones(batch_size, device=device) # real label = 1

        # generated (fake) images
        noise = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)  # format NCHW
        fake_images = model.generator_forward(noise)
        print(f"Fake Images Shape: {fake_images.shape}")
        fake_labels = torch.zeros(batch_size, device=device) # fake label = 0
        flipped_fake_labels = real_labels # here, fake label = 1


        # --------------------------
        # Train Discriminator
        # --------------------------

        optim_discr.zero_grad()

        # Before loss calculation
        # print("Discriminator real predictions shape:", discr_pred_real.shape)
        # print("Discriminator fake predictions shape:", discr_pred_fake.shape)


        # get discriminator loss on real images
        discr_pred_real = model.discriminator_forward(real_images).view(-1) 
        real_score = torch.sigmoid(discr_pred_real).mean().item()
        real_scores.append(real_score)
        real_loss = F.binary_cross_entropy_with_logits(discr_pred_real, real_labels)
        # real_loss.backward()

        # get discriminator loss on fake images
        discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
        fake_score = torch.sigmoid(discr_pred_fake).mean().item()
        fake_scores.append(fake_score)
        fake_loss = F.binary_cross_entropy_with_logits(discr_pred_fake, fake_labels)

        # combined loss
        discr_loss = 0.5*(real_loss + fake_loss)
        d_losses.append(discr_loss)
        discr_loss.backward()

        optim_discr.step()

        # --------------------------
        # Train Generator
        # --------------------------

        optim_gen.zero_grad()

        # get discriminator loss on fake images with flipped labels
        discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
        gener_loss = F.binary_cross_entropy_with_logits(discr_pred_fake, flipped_fake_labels)
        g_losses.append(gener_loss)
        gener_loss.backward()

        optim_gen.step()

        # Adjust learning rate if lr_scheduler is provided
        # if lr_scheduler is not None:
            # lr_scheduler.step()

        # --------------------------
        # Logging
        # --------------------------   
        log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
        log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
        predicted_labels_real = torch.where(discr_pred_real > 0., 1., 0.)
        predicted_labels_fake = torch.where(discr_pred_fake > 0., 1., 0.)
        acc_real = (predicted_labels_real == real_labels).float().mean().item() * 100.0
        acc_fake = (predicted_labels_fake == fake_labels).float().mean().item() * 100.0
        log_dict['train_discriminator_real_acc_per_batch'].append(acc_real)
        log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake) 
        real_accs.append(acc_real)
        fake_accs.append(acc_fake)

            
        # epoch_real_acc += acc_real
        # epoch_fake_acc += acc_fake

        # print batch loss, discriminator scores, and accuracy
        print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {batch_idx+1}/{len(train_loader)}] "
              f"[D loss: {discr_loss:.6f}] [G loss: {gener_loss:.6f}] "
              f"[D's scores on real images: {real_score:.6f}] [fake score: {fake_score:.6f}] "
              f"[D's accuracies on real images: {acc_real:.2%}] [fake images: {acc_fake:.2%}]")

    # calculate average loss and scores for the current epoch
    print(d_losses)
    epoch_d_loss = sum(d_losses) / len(d_losses)
    epoch_g_loss = sum(g_losses) / len(g_losses)
    epoch_real_score = sum(real_scores) / len(real_scores)
    epoch_fake_score = sum(fake_scores) / len(fake_scores)
    epoch_real_acc = sum(real_accs)/len(real_accs)
    epoch_fake_acc = sum(fake_accs)/len(fake_accs)
    epoch_real_acc /= num_batches
    epoch_fake_acc /= num_batches

    # store epoch metrics data
    all_d_losses.append(epoch_d_loss)
    all_g_losses.append(epoch_g_loss)
    all_real_scores.append(epoch_real_score)
    all_fake_scores.append(epoch_fake_score)
    all_real_accs.append(epoch_real_acc)
    all_fake_accs.append(epoch_fake_acc)

    # save past data
    if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(), 
            'discriminator_state_dict': model.discriminator.state_dict(),  
            'optimizer_G_state_dict': optim_gen.state_dict(),  
            'optimizer_D_state_dict': optim_discr.state_dict(),  
            'lossG': gener_loss.item,  
            'lossD': discr_loss.item,  
            'real_score': epoch_real_score,       
            'fake_score': epoch_fake_score,     
            'real_acc': epoch_real_acc,            
            'fake_acc': epoch_fake_acc            
        }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))

    # output training stats for the epoch
    print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {batch_idx+1}/{len(train_loader)}] "
            f"[D loss: {epoch_d_loss:.6f}] [G loss: {epoch_g_loss:.6f}] "
            f"[D's epoch mean scores on real images: {epoch_real_score:.6f}] [fake images: {epoch_fake_score:.6f}] "
            f"[D's epoch mean accuracies on real images: {epoch_real_acc:.2%}] [fake images: {epoch_fake_acc:.2%}]")

        # end timer for epoch
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60.0
    print(f"Epoch {epoch+1} took {elapsed_minutes:.2f} minutes.")

    model.eval() 
    with torch.no_grad():
      if (epoch + 1) <=900:
        noise = torch.randn(5, opt.latent_dim, 1, 1).to(device) 
        fake = model.generator_forward(noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        plt.axis('off')
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.savefig(os.path.join("reportR2/images", f"epoch_{epoch+1}_generated_images.png"))
        plt.close()
        model.train() 
      else:
        noise = torch.randn(100, opt.latent_dim, 1, 1).to(device) 
        fake = model.generator_forward(noise).detach().cpu()
        img_grid = torchvision.utils.make_grid(fake, padding=2, normalize=True)
        plt.axis('off')
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.savefig(os.path.join("reportR2/images", f"epoch_{epoch+1}_generated_images.png"))
        plt.close()
        model.train() 


    directoryL = "reportR2/losses"
    directoryS = "reportR2/scores"
    directoryA = "reportR2/accuracies"
    if not os.path.exists(directoryL):
        os.makedirs(directoryL)
    if not os.path.exists(directoryA):
        os.makedirs(directoryA)
    if not os.path.exists(directoryS):
        os.makedirs(directoryS)
    all_d_losses_cpu = []
    all_g_losses_cpu = []
    all_real_accs_cpu = []
    all_fake_accs_cpu = []
    all_real_scores_cpu = []
    all_fake_scores_cpu = []

    if (epoch + 1) % 30 == 0 or (epoch + 1) == 1:
        # Make sure data is on the CPU and in a suitable format (numpy array or list)
        for loss in all_d_losses:
            if isinstance(loss, torch.Tensor):  # Check if the element is a tensor
                if loss.is_cuda:
                    loss = loss.cpu()  # Move to CPU if it's on CUDA
                loss = loss.detach().numpy()  # Convert to numpy array
            all_d_losses_cpu.append(loss)
        for loss in all_g_losses:
            if isinstance(loss, torch.Tensor):  # Check if the element is a tensor
                if loss.is_cuda:
                    loss = loss.cpu()  # Move to CPU if it's on CUDA
                loss = loss.detach().numpy() # Convert to numpy array
            all_g_losses_cpu.append(loss)

        for acc in all_real_accs:
            if isinstance(acc, torch.Tensor):  # Check if the element is a tensor
                if acc.is_cuda:
                    acc = acc.cpu()  # Move to CPU if it's on CUDA
                acc = acc.detach().numpy()  # Convert to numpy array
            all_real_accs_cpu.append(acc)

        for acc in all_fake_accs:
            if isinstance(acc, torch.Tensor):  # Check if the element is a tensor
                if acc.is_cuda:
                    acc = acc.cpu()  # Move to CPU if it's on CUDA
                acc = acc.detach().numpy()  # Convert to numpy array
            all_fake_accs_cpu.append(acc)

        for score in all_real_scores:
            if isinstance(score, torch.Tensor):  # Check if the element is a tensor
                if score.is_cuda:
                    score = score.cpu()  # Move to CPU if it's on CUDA
                score = score.detach().numpy()  # Convert to numpy array
            all_real_scores_cpu.append(score)

        for score in all_fake_scores:
            if isinstance(score, torch.Tensor):  # Check if the element is a tensor
                if score.is_cuda:
                    score = score.cpu()  # Move to CPU if it's on CUDA
                score = score.detach().numpy() # Convert to numpy array
            all_fake_scores_cpu.append(score)

        # losses graph
        plt.figure(figsize=(10, 5))
        plt.plot(all_d_losses_cpu, label='Discriminator Loss')
        plt.plot(all_g_losses_cpu, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator and Generator Losses')
        plt.legend()
        plt.savefig(os.path.join("reportR2/losses", f"losses_graph_epoch_{epoch+1}.png"))
        plt.close()

        # accuracies graph
        plt.figure(figsize=(10, 5))
        plt.plot(all_real_accs_cpu, label='Real Image Accuracy')
        plt.plot(all_fake_accs_cpu, label='Fake Image Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.title("Discriminator's Accuracies of Real and Fake Images")
        plt.legend()
        plt.savefig(os.path.join("reportR2/accuracies", f"accuracy_graph_epoch_{epoch+1}.png"))
        plt.close()

        # scores
        plt.figure(figsize=(10, 5))
        plt.plot(all_real_scores_cpu, label='Real Image Scores')
        plt.plot(all_fake_scores_cpu, label='Fake Image Scores')
        plt.xlabel('Epochs')
        plt.ylabel('Scores')
        plt.title('Discriminator Scores on real and fake images')
        plt.legend()
        plt.savefig(os.path.join("reportR2/scores", f"scores_graph_epoch_{epoch + 1}.png"))
        plt.close()
