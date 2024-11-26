import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from torchvision.models import ResNet50_Weights, resnet50
#-------------------------|
#        Hyperparam
#-------------------------|
batch_size = 32
#-------------------------|

# dir + gpu support
image_dir = 'train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#transformations in order to preprocess images using resnet50 standards
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageFeatureExtractionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
#exclude xml, since it was handled earlier
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, image_name



dataset = ImageFeatureExtractionDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#                                               Pre-trained ResNet50 model
weights = ResNet50_Weights.IMAGENET1K_V1
resnet_model = resnet50(weights=weights)
#remove classification head
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

resnet_model = resnet_model.to(device)
resnet_model.eval()

#                                           Store feature vectors in the list
features_list = []
with torch.no_grad():
    for images, filenames in dataloader:
        images = images.to(device)
        features = resnet_model(images)
        features = features.view(features.size(0), -1)  #normalize
        features_list.extend(features.cpu().numpy())

features_array = np.array(features_list)
np.save('extracted_features.npy', features_array)

print("Feature extraction complete!\n Features saved as 'extracted_features.npy'.")
