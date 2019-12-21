import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse
import json



parser = argparse.ArgumentParser(description='Inference for classification')
parser.add_argument('-i','--image_path',type=str, metavar='', required=True, help='path to image to predict e.g. flowers/test/class/image')
parser.add_argument('-t','--top_k', type=int, metavar='', default=1, help='print out the top K classes along with associated probabilities')
parser.add_argument('-c','--category_names', type=str, metavar='', default='cat_to_name.json', help='load a JSON file that maps the class values to other category names')
parser.add_argument('-g','--gpu',action="store_true", default=False, help='choose training the model on a GPU')

args = parser.parse_args()






with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
     # Load the saved file
    checkpoint = torch.load("checkpoint.pth")
    architecture = checkpoint['architecture']
    # Download pretrained model
    model = getattr(models, architecture)(pretrained=True);
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
        
    hidden_units = checkpoint['hidden_units']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer = optim.Adam(model.classifier.parameters())
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    
    
    return model

model = load_checkpoint('checkpoint.pth')


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    size = 256,256
    im.thumbnail(size)
    #Crop
    left = (256-224)/2
    top = (256-224)/2
    right = (left + 224)
    bottom = (top + 224)
    im = im.crop((left, top, right, bottom))
   
    np_image = np.array(im)
    np_image = np_image / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose(2, 0, 1)
    return np_image

# Reversing idx to class
idx_to_class = {}
for key, value in model.class_to_idx.items():
    idx_to_class[value] = key


def predict(image_path, model, topk):	
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use GPU if it's available
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # TODO: Implement the code to predict the class from an image file
    #preprocess image
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    
    # turn off dropout
    model.eval()

    #Load image and the model to cpu or gpu
    model.to(device)
    image = image.to(device)

    logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
  
    top_class = np.array(top_class)[0]
    top_p = np.array(top_p.detach())[0]
    
    # Mapping index to class
    top_classes = []
    for i in range(len(top_class)):
        top_classes.append(idx_to_class[top_class[i]])
    
    # Mapping class to flower name
    flower_names = []
    for i in range(len(top_classes)):
        flower_names.append(cat_to_name[top_classes[i]])
    
    return top_p, flower_names


probs, classes = predict(args.image_path, model, args.top_k)

print(f"class Probability: {probs}")
print(f"flower name: {classes}")
