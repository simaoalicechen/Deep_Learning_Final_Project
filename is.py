import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from numpy import expand_dims, log, mean, exp

image_dir = 'reportCW/images'

# custom dataset to process and slice out the images from the generated images that were
# saved to each report directory
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')  

        # TODO, hardcoded the column and rows, need to change according to how the images were generated
        
        num_cols = 8
        num_rows = 2
        sub_images = []

        if self.transform:
            width, height = image.size
            sub_image_width = width // num_cols
            sub_image_height = height // num_rows
            
            # slice the sub images out of the whole png
            # left is the most left pixel index
            for i in range(num_rows):
                for j in range(num_cols):
                    left = j * sub_image_width
                    top = i * sub_image_height
                    right = left + sub_image_width
                    bottom = top + sub_image_height
                    sub_image = image.crop((left, top, right, bottom))

                    sub_image = self.transform(sub_image)
                    sub_images.append(sub_image)

        return torch.stack(sub_images) 


# def load_images(root_dir, transform):
#     dataset = CustomDataset(root_dir, transform=transform)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     return loader

# def preprocess_image():
transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # return transform

def get_inception_model():
  # add the weights from inception_V3_Weights, to avoid loading error
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=True)
    model.eval() 
    return model

# source: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/#:~:text=The%20calculation%20of%20the%20inception%20score%20on%20a%20group%20of,group%20(p(y)).

def calculate_inception_score(p_yx, eps=1E-16):
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = mean(sum_kl_d)
    is_score = exp(avg_kl_d)
    return is_score

if __name__ == '__main__':
    image_dir = 'reportCW/images'  
    transform = transform
    image_loader = DataLoader(CustomDataset(image_dir, transform), batch_size=1, shuffle=False)
    model = get_inception_model()

    for images in image_loader:
        img = images.squeeze(0)
        # img = img.unsqueeze(0)   
        with torch.no_grad():
            logits = model(img)
            probs = torch.nn.functional.softmax(logits, dim=1)
            p_yx = probs.detach().cpu().numpy() 
        score = calculate_inception_score(p_yx)
        print(f"IS: {score}")



