import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

#-------------------------|
#        Hyperparam
#-------------------------|
learning_rate = 0.01
batch_size = 32
total_epochs = 50
output_features = 5
test_folder = "test"
csv_file = "test.csv"
#-------------------------|



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
#ensure output_features = training output_features
model.fc = torch.nn.Linear(num_ftrs, output_features)
model = model.to(device)

model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

extracted_features = np.load('extracted_features.npy', allow_pickle=True)

# Transformations of images
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_df = pd.read_csv(csv_file)

#Maping classes to types
class_to_taxonomy = {
    0: {'Family': 'Amaryllidaceae', 'Genus': 'Narcissus', 'Species': 'Narcissus dubius Gouan'},
    1: {'Family': 'Asteraceae', 'Genus': 'Helianthus', 'Species': 'Helianthus annuus'},
    2: {'Family': 'Fabaceae', 'Genus': 'Medicago', 'Species': 'Medicago sativa'},
    3: {'Family': 'Rosaceae', 'Genus': 'Rosa', 'Species': 'Rosa gallica'},
    4: {'Family': 'Solanaceae', 'Genus': 'Solanum', 'Species': 'Solanum lycopersicum'},
}

#                                                   Predictions
results = []
for _, row in test_df.iterrows():
    media_id = row['MediaId']
    image_path = os.path.join(test_folder, f"{media_id}.jpg")

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class_id = predicted.item()

            # Map the predicted class ID to taxonomic info
            taxonomy = class_to_taxonomy.get(predicted_class_id, {})
            family = taxonomy.get('Family', '')
            genus = taxonomy.get('Genus', '')
            species = taxonomy.get('Species', '')

        results.append({
            "ObservationId": row.get("ObservationId", ""),
            "MediaId": media_id,
            "Vote": row.get("Vote", ""),
            "Prediction": predicted_class_id,
            "Family": family,
            "Genus": genus,
            "Species": species
        })

#                                 save results to a csv file

results_df = pd.DataFrame(results)
results_df.to_csv("Test_ResultsV1.csv", index=False)
print("Classification completed. Results saved to 'test_resultsV1.csv'.")

