import torch
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import torch.nn as nn

model_path = 'ayame_model_1.pt'
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 2
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(model_state_dict)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_image(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    classes = ['ayame', 'not ayame']
    prediction = classes[predicted[0].item()]

    return prediction


image_input = gr.inputs.Image()
label_output = gr.outputs.Label(num_top_classes=2)
examples = [
    ['./sample/Ayame.png'], ['./sample/Nene.png'], ['./sample/Miko.png'], ['./sample/Towa.png'], ['./sample/Lamy.png'],
    ['./sample/Iroha.png']
]
gr.Interface(fn=predict_image, inputs=image_input, outputs=label_output, title='IsAyame?', examples=examples,
             layout="vertical", output_type='auto').launch()
