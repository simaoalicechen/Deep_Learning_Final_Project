# ditched the previously defined Inception Score function to use the ignite.metrics' InceptionScore 
# method, because it requires less data processing

import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from torchmetrics.image.fid import FrechetInceptionDistance
from ignite.utils import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

#load images so that each image can be individually extracted and then processed later
# each png file probably has more than one image depending on how each model saved the file
# in the past
def load_images_from_file(file_path, image_count=64):
    img = Image.open(file_path).convert('RGB')  
    img_width, img_height = img.size
    single_width = img_width // 8
    single_height = img_height // 8
    images = []
    # TODO, needs to change the col and row, depending on how each model output their images
    for i in range(8):  
        for j in range(8): 
            box = (j * single_width, i * single_height, (j + 1) * single_width, (i + 1) * single_height)
            images.append(img.crop(box).resize((299, 299)))  
    return images

# transform each extracted image from the epoch batchs one by one
metric = InceptionScore(output_transform=lambda x: x, device="cuda" if torch.cuda.is_available() else "cpu")
metric_2 = FrechetInceptionDistance(feature=64, normalize=True)
data_root = 'data/celebA'
dataset_folder = f'{data_root}'
transform = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.CenterCrop((299, 299)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])
dataset_obj = datasets.ImageFolder(data_root, transform)
subset_obj = torch.utils.data.Subset(dataset_obj, list(range(0, len(dataset_obj))))
dataloader = torch.utils.data.DataLoader(subset_obj,
    batch_size = 16,
    shuffle=True,
)
iterator_dl = iter(dataloader)
# metric_2 = FrechetInceptionDistance(feature=64, normalize=True)
directory_path = 'report_DCGAN64_CelebA/TestsFinal'
average_scores = []
average_scores_2 = []

# process each file
for file_name in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, file_name)
    if file_path.endswith('.png'):
        images = load_images_from_file(file_path)
        image_tensors = [torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().div(255) for image in images]
        image_tensors = torch.cat(image_tensors, dim=0) 
        data_loader = torch.utils.data.DataLoader(image_tensors, batch_size=64)
        # need to reset to calculate the score per epoch 
        metric.reset()
        metric_2.reset()
        for images in data_loader:
            metric.update(images) 
            metric_2.update(images, real=False)
            metric_2.update(next(iterator_dl)[0], real=True)
        score = metric.compute()
        score_2 = metric_2.compute()
        # all_scores.append(score)
        average_scores.append(score)
        average_scores_2.append(score_2)
        print("checking average scores", len(average_scores))
        print("checking average scores 2", len(average_scores_2))

# statistical analysis on all collected scores
scores_array = np.array(average_scores)
median_score = np.median(average_scores)
mean_score = np.mean(average_scores)
std_deviation = np.std(average_scores)
top_10_percent = np.percentile(average_scores, 90)
lowest_10_percent = np.percentile(average_scores, 10)


print(f"Median of All Inception Scores: {median_score}")
print(f"Mean of All Inception Scores: {mean_score}")
print(f"Standard Deviation of All Scores: {std_deviation}")
print(f"Top 10% of All Inception Scores: {top_10_percent}")
print(f"Lowest 10% of All Inception Scores: {lowest_10_percent}")

# dir
output_dir = 'report_DCGAN64_CelebA/ISScore'
os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, 'output_statistics.txt')
with open(output_file_path, 'w') as f:
    print(f"Median of All Inception Scores: {median_score}", file=f)
    print(f"Mean of All Inception Scores: {mean_score}", file=f)
    print(f"Standard Deviation of All Scores: {std_deviation}", file=f)
    print(f"Top 10% of All Inception Scores: {top_10_percent}", file=f)
    print(f"Lowest 10% of All Inception Scores: {lowest_10_percent}", file=f)

print(f"Statistics have been saved to '{output_file_path}'")
# plot and save the graph

plt.figure(figsize=(10, 5))
plt.plot(average_scores, marker='o')
plt.title('Trend of Inception Scores')
plt.xlabel('PNG File Index')
plt.ylabel('Average Inception Score')
plt.grid(True)

plot_filename = os.path.join(output_dir, 'inception_score_trend.png')
plt.savefig(plot_filename)
plt.close()

print(f"plot saved to {plot_filename}")
# statistical analysis on all collected scores
scores_array_2 = np.array(average_scores_2)
median_score_2 = np.median(average_scores_2)
mean_score_2 = np.mean(average_scores_2)
std_deviation_2 = np.std(average_scores_2)
top_10_percent_2 = np.percentile(average_scores_2, 90)
lowest_10_percent_2 = np.percentile(average_scores_2, 10)

print(f"Median of All FID Scores: {median_score_2}")
print(f"Mean of All FID Scores: {mean_score_2}")
print(f"Standard Deviation of All FID Scores: {std_deviation_2}")
print(f"Top 10% of All FID Scores: {top_10_percent_2}")
print(f"Lowest 10% of All FID Scores: {lowest_10_percent_2}")

# dir
output_dir = 'report_DCGAN64_CelebA/ISScore'
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, 'output_statistics_2.txt')
with open(output_file_path, 'w') as f:
    print(f"Median of All FID Scores: {median_score_2}", file=f)
    print(f"Mean of All FID Scores: {mean_score_2}", file=f)
    print(f"Standard Deviation of All FID Scores: {std_deviation_2}", file=f)
    print(f"Top 10% of All FID Scores: {top_10_percent_2}", file=f)
    print(f"Lowest 10% of All FID Scores: {lowest_10_percent_2}", file=f)
print(f"Statistics have been saved to '{output_file_path}'")

# plot and save the graph
plt.figure(figsize=(10, 5))
plt.plot(average_scores_2, marker='o')
plt.title('Trend of FID Scores')
plt.xlabel('PNG File Index')
plt.ylabel('Average FID Score')
plt.grid(True)

plot_filename = os.path.join(output_dir, 'fid_score_trend.png')
plt.savefig(plot_filename)
plt.close()
print(f"plot saved to {plot_filename}")
