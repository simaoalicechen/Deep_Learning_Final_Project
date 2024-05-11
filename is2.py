# basically the same thing, except made some custom adjustment, as in reportR2, the images' grid sizes changed after certain epochs
import torch
from ignite.metrics import InceptionScore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_images_from_file(file_path, index, transition_index=112):
    img = Image.open(file_path).convert('RGB')  
    img_width, img_height = img.size
    
    if index <= transition_index:
        rows, cols = 2, 8  
    else:
        rows, cols = 8, 8 
    
    single_width = img_width // cols
    single_height = img_height // rows
    images = []
    for i in range(rows):
        for j in range(cols):
            box = (j * single_width, i * single_height, (j + 1) * single_width, (i + 1) * single_height)
            images.append(img.crop(box).resize((299, 299))) 
    return images

metric = InceptionScore(output_transform=lambda x: x, device="cuda" if torch.cuda.is_available() else "cpu")

directory_path = 'reportR2/images'

average_scores = []

files = sorted(os.listdir(directory_path))

for index, file_name in enumerate(files):
    file_path = os.path.join(directory_path, file_name)
    if file_path.endswith('.png'):
        images = load_images_from_file(file_path, index)
        
        image_tensors = [torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().div(255) for image in images]
        image_tensors = torch.cat(image_tensors, dim=0) 
        
        data_loader = torch.utils.data.DataLoader(image_tensors, batch_size=min(len(image_tensors), 16)) 
        
        for images in data_loader:
            metric.update((images)) 
        
        score = metric.compute()
        average_scores.append(score)
        print("checking ", len(average_scores))

output_dir = 'reportR2/ISScore'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(average_scores, marker='o')
plt.title('Trend of Inception Scores')
plt.xlabel('PNG File Index')
plt.ylabel('Average Inception Score')
plt.grid(True)
# Save the plot
plot_filename = os.path.join(output_dir, 'inception_score_trend.png')
plt.savefig(plot_filename)
plt.close()

print(f"Plot saved to {plot_filename}")
