import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from numpy import expand_dims, log, mean, exp
import matplotlib.pyplot as plt

# image_dir = 'reportR/images'

# custom dataset to process and slice out the images from the generated images that were
# saved to each report directory
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))])
        self.num_columns = 8
        self.num_rows = 2

    def __len__(self):
        # Each image is split into 16 sub-images, so the dataset size is multiplied
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        width, height = image.size
        sub_image_width = width // self.num_columns
        sub_image_height = height // self.num_rows
        sub_images = []

        for i in range(self.num_rows):
            for j in range(self.num_columns):
                left = j * sub_image_width
                top = i * sub_image_height
                right = left + sub_image_width
                bottom = top + sub_image_height
                sub_image = image.crop((left, top, right, bottom))
                
                sub_images.append(sub_image)

        return sub_images, self.images[idx]

# this function makes sure the list of subimages extracted from each png file 
# will be able to be processed by the Dataloader later.
def custom_collate(batch):
    # `batch` is a list where each element is the output from `CustomDataset.__getitem__`
    sub_images, filenames = zip(*batch)
    # flatten the list of lists into a single list if your batch size > 1
    sub_images = [item for sublist in sub_images for item in sublist]
    return sub_images, filenames

def get_inception_model():
  # add the weights from inception_V3_Weights, to avoid loading error
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=True)
    model.eval() 
    return model

# source: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/#:~:text=The%20calculation%20of%20the%20inception%20score%20on%20a%20group%20of,group%20(p(y)).

def calculate_inception_score(p_yx, eps=1E-16):
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)  # Calculate the marginal distribution
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))  # Compute KL divergence for each image's distribution
    sum_kl_d = kl_d.sum(axis=1)  # Sum the KL divergences over classes for each image
    avg_kl_d = np.mean(sum_kl_d)  # Average these sums over all images
    is_score = np.exp(avg_kl_d)  # Take the exponential of the average KL divergence
    return is_score


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    title = "DCGANCelebA64"
    image_dir = 'reportR/images'
    image_loader = DataLoader(CustomDataset(image_dir), batch_size=1, shuffle=False, collate_fn=custom_collate)
    model = get_inception_model()

    # all_scores = []
    average_scores = []
    for sub_images, filename in image_loader:
        scores = []
        for sub_image in sub_images:
            sub_image = transform(sub_image)
            with torch.no_grad():
                logits = model(sub_image.unsqueeze(0))
                probs = torch.nn.functional.softmax(logits, dim=1)
                p_yx = probs.detach().cpu().numpy()
            score = calculate_inception_score(p_yx)
            # print(score)
            scores.append(score)
            print(scores)

        average_score = np.mean(scores)
        average_scores.append(average_score)
        print(f'Average Inception Score for {filename}: {average_score}')
