import torchvision
import torch.nn as nn
import torch 
import numpy as np
from source.Model import Model
from torchvision import transforms,models,datasets
from PIL import Image

model = torch.load("./weights/Model-epoch 3pkl.pth")

def transform_image(infile):
    input_transforms = [transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg

def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    print("inference logits",outputs)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction

if __name__:
    file = "./dataset/passport/passport.jpg"
    input_tensor = transform_image(file)
    prediction_idx = get_prediction(input_tensor)
    print(prediction_idx)

