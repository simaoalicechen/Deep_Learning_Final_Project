# ditched the previously defined Inception Score function to use the ignite.metrics' InceptionScore 
# method, because it requires less data processing

import torch
from ignite.metrics import InceptionScore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
#

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

directory_path = 'report_DCGAN128_CelebA/tests'
average_scores = []
# all_scores  = []
# process each file
for file_name in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, file_name)
    if file_path.endswith('.png'):
        images = load_images_from_file(file_path)
        image_tensors = [torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().div(255) for image in images]
        image_tensors = torch.cat(image_tensors, dim=0) 
        data_loader = torch.utils.data.DataLoader(image_tensors, batch_size=16)
        # need to reset to calculate the score per epoch 
        metric.reset()
        for images in data_loader:
            metric.update(images) 
        score = metric.compute()
        # all_scores.append(score)
        average_scores.append(score)
        print("checking avergae scores", len(average_scores))

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
output_dir = 'report_DCGAN128_CelebA/ISScore'
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
