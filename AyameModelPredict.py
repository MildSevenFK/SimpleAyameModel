import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

parser = argparse.ArgumentParser(description='Predict the character in the image.')
parser.add_argument('image_path', type=str, help='path to the image file')
parser.add_argument('model_path', type=str, help='path to the saved model file')
parser.add_argument('--similarity', action='store_true', help='whether to output similarity scores')
args = parser.parse_args()

data_path = 'data path'

dataset = ImageFolder(data_path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

model = torch.load(args.model_path, map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(args.image_path).convert('RGB')
img = transform(img)
img = img.unsqueeze(0)

with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)

classes = ['ayame', 'no_ayame']
prediction = classes[predicted[0].item()]

if args.similarity:
    similarities = []
    for i in range(len(dataset)):
        similarity = torch.nn.functional.cosine_similarity(model(img),
                                                           model(dataset[i][0].unsqueeze(0))).item()
        similarities.append(similarity)
    similarity = max(similarities)
    print(f"Predicted character: {prediction}")
    print(f"Similarity: {similarity}")
else:
    print(f"Predicted character: {prediction}")