import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

#-------------------------|
#        Hyperparam
#-------------------------|
learning_rate = 0.01
batch_size = 32
total_epochs = 50
output_features = 5
#-------------------------|


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# label encoders for categorical columns & ensure label encoders only on the respective columns
class ImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.vote_encoder = LabelEncoder()
        self.class_id_encoder = LabelEncoder()
        self.family_encoder = LabelEncoder()
        self.genus_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()

        self.vote_encoder.fit(self.df['Vote'].dropna().unique())
        self.class_id_encoder.fit(self.df['ClassId'].dropna().unique())
        self.family_encoder.fit(self.df['Family'].dropna().unique())
        self.genus_encoder.fit(self.df['Genus'].dropna().unique())
        self.species_encoder.fit(self.df['Species'].dropna().unique())

    def __len__(self):
        return len(self.df)

    # MediaId is index 1
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.df.iloc[idx, 1]}.jpg")
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        #                            Extract and encode the labels
        #re-done becuase of errors with the npy extracted

        vote = self.vote_encoder.transform([self.df.iloc[idx, 2]])[0] if not pd.isna(
            self.df.iloc[idx, 2]) else -1
        class_id = self.class_id_encoder.transform([self.df.iloc[idx, 4]])[0] if not pd.isna(
            self.df.iloc[idx, 4]) else -1
        family = self.family_encoder.transform([self.df.iloc[idx, 5]])[0] if not pd.isna(self.df.iloc[idx, 5]) else -1
        genus = self.genus_encoder.transform([self.df.iloc[idx, 6]])[0] if not pd.isna(self.df.iloc[idx, 6]) else -1
        species = self.species_encoder.transform([self.df.iloc[idx, 7]])[0] if not pd.isna(self.df.iloc[idx, 7]) else -1

        labels = [vote, class_id, family, genus, species]

        return image, torch.tensor(labels, dtype=torch.float32)


# Transformations of images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#load data
train_dataset = ImageDataset(csv_file='train.csv', image_dir='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#                                           Model used is pretrained resnet50

model = models.resnet50(weights='IMAGENET1K_V1')

# Desired features are 5: (5 labels: Vote, ClassId, Family, Genus, Species)
model.fc = nn.Linear(model.fc.in_features, output_features)

model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{total_epochs}', ncols=100):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {running_loss / len(train_dataloader)}")


#save model

torch.save(model.state_dict(), 'model.pth')
print("Model saved as 'model.pth'")
