import torch
# Import your model classes, adjust the module names as per your project structure
# from cw import Generator, Discriminator  
from torch.optim import Adam  # or whatever optimizer you are using
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def load_checkpoint(model_path, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    lossG = checkpoint['lossG']
    lossD = checkpoint['lossD']
    real_score = checkpoint['real_score']
    fake_score = checkpoint['fake_score']
    real_acc = checkpoint['real_acc']
    fake_acc = checkpoint['fake_acc']
    return start_epoch, lossG, lossD, real_score, fake_score, real_acc, fake_acc