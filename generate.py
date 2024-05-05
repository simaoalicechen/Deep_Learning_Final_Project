import torch
import ssl
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

celeba_dataset = datasets.CelebA(root='data/celebA', split='all', transform=transform, download=True)

batch_size = 64
dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)

use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo', 'DCGAN', pretrained=True, useGPU=use_gpu)

num_images = 64
noise, _ = model.buildNoiseData(num_images)

with torch.no_grad():
    generated_images = model.test(noise, True, 0)

# use cpu
generated_images_cpu = generated_images.cpu()

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(
    np.transpose(vutils.make_grid(generated_images_cpu, padding=2, normalize=True).cpu().numpy(), (1, 2, 0))
)
plt.savefig("generated_image.jpg")
