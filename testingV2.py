import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image

# Mapping ID to types
df = pd.read_csv('train.csv')
class_to_family = df[['ClassId', 'Family']].drop_duplicates().set_index('ClassId')['Family'].to_dict()
class_to_genus = df[['ClassId', 'Genus']].drop_duplicates().set_index('ClassId')['Genus'].to_dict()
class_to_species = df[['ClassId', 'Species']].drop_duplicates().set_index('ClassId')['Species'].to_dict()

class_id_mapping = {idx: class_id for idx, class_id in enumerate(df['ClassId'].unique())}

#                                        make predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# Load extracted features
extracted_features = np.load('extracted_features.npy', allow_pickle=True)

# Transformations of images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#                                           Process and predict
test_data = pd.read_csv('test.csv')
results = []
for index, row in test_data.iterrows():
    media_id = row['MediaId']
    image_path = f'test/{media_id}.jpg'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = outputs.argmax(dim=1).item()
        predicted_class = class_id_mapping[predicted_index]

#get mappings
    family = class_to_family.get(predicted_class, "Unknown")
    genus = class_to_genus.get(predicted_class, "Unknown")
    species = class_to_species.get(predicted_class, "Unknown")

    results.append({
        "MediaId": media_id,
        "ClassId": predicted_class,
        "Family": family,
        "Genus": genus,
        "Species": species,
    })


output_df = pd.DataFrame(results)
output_df.to_csv('test_resultsV2.csv', index=False)

print("Classification completed. Results saved to 'test_resultsV2.csv'.")
