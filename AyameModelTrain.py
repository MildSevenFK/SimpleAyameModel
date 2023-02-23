import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

data_path = "datapath"
save_path = "savepath/my_model_{}.pt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = len(dataset.classes)
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

last_layer_name = 'fc.weight'
last_layer = None
for name, layer in model.named_parameters():
    if name == last_layer_name:
        last_layer = layer

def train_and_save(model, dataloader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    for epoch in range(1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print('Epoch {} [{}/{}] loss: {}'.format(epoch + 1, i + 1, len(dataloader), running_loss / 10))
                running_loss = 0.0

        torch.save(model.state_dict(), save_path.format(epoch + 1))

    print("Training complete!")

train_and_save(model, dataloader, save_path)