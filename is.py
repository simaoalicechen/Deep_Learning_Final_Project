# ditched the previously defined Inception Score function to use the ignite.metrics' InceptionScore 
# method, because it requires less data processing

import torch
from ignite.metrics import InceptionScore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
#

#load images so that each image can be individually extracted and then processed later
# each png file probably has more than one image depending on how each model saved the file
# in the past
def load_images_from_file(file_path, image_count=16):
    img = Image.open(file_path).convert('RGB')  
    img_width, img_height = img.size
    single_width = img_width // 8
    single_height = img_height // 2
    images = []
    # TODO, needs to change the col and row, depending on how each model output their images
    for i in range(2):  
        for j in range(8): 
            box = (j * single_width, i * single_height, (j + 1) * single_width, (i + 1) * single_height)
            images.append(img.crop(box).resize((299, 299)))  
    return images

# transform each extracted image from the epoch batchs one by one
metric = InceptionScore(output_transform=lambda x: x, device="cuda" if torch.cuda.is_available() else "cpu")

directory_path = 'reportR128/images'
average_scores = []
# process each file
for file_name in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, file_name)
    if file_path.endswith('.png'):
        images = load_images_from_file(file_path)
        
        image_tensors = [torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().div(255) for image in images]
        image_tensors = torch.cat(image_tensors, dim=0) 
        
        data_loader = torch.utils.data.DataLoader(image_tensors, batch_size=16)
        
        for images in data_loader:
            metric.update(images) 
        
        score = metric.compute()
        average_scores.append(score)
        print("checking ", len(average_scores))


# dir
output_dir = 'reportR128/ISScore'
os.makedirs(output_dir, exist_ok=True)

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
