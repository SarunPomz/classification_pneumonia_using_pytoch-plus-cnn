import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

def classify(image, model, class_names):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence = probabilities.max().item() * 100
        class_name = class_names[probabilities.argmax().item()]
    
    return {"class": class_name, "confidence": confidence}