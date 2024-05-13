import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('saved_models/checkpoint.pth', map_location=torch.device('cpu'))

gen_loss = checkpoint['g_loss']
dis_loss = checkpoint['d_loss']

np_gen = []
np_dis = []
temp_empty = []

loss_per_epoch = np.round(len(gen_loss)/90)

for i, gen in enumerate(gen_loss):
    temp_empty.append(gen.detach().numpy())
    if i % loss_per_epoch == 0:
        np_gen.append(np.average(temp_empty))
        temp_empty = []
temp_empty = []
for i, dis in enumerate(dis_loss):
    temp_empty.append(dis.detach().numpy())
    if i % loss_per_epoch == 0:
        np_dis.append(np.average(temp_empty))
        temp_empty = []

plt.figure(figsize=(10, 5))
plt.plot(np_dis, label="Discriminator Loss")
plt.plot(np_gen, label="Generator Loss")
plt.title('Discriminator and Generator Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("rel_loss.png")
