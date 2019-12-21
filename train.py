import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
import time
from workspace_utils import active_session
from collections import OrderedDict
import numpy as np
from PIL import Image




import argparse

parser = argparse.ArgumentParser(description='Building and training the classifier')
parser.add_argument('-a','--arch', type=str, metavar='', default="vgg16", help='choose from any of these model architectures available from torchvision.models [densenet121, vgg13, vgg16]')
parser.add_argument('-l','--learning_rate', type=float, metavar='', default=0.003, help='set hyperparameters for learning rate')
parser.add_argument('-hid','--hidden_units', type=int, metavar='', default=1024, help='set hyperparameters for number of hidden units')
parser.add_argument('-e','--epochs', type=int, metavar='', default=5, help='set hyperparameters for training epochs')
parser.add_argument('-g','--gpu',action="store_true", default=False, help='choose training the model on a GPU')
args = parser.parse_args()



data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#  Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#  Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
 

#  Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



# Load a pre-trained model 
if args.arch == 'densenet121':
    print(f"Using {args.arch} as a pretrained model")
    model = models.densenet121(pretrained=True)
    input_size = model.classifier.in_features
elif args.arch == 'vgg13':
    print(f"Using {args.arch} as a pretrained model")
    model = models.vgg13_bn(pretrained=True)
    input_size = model.classifier[0].in_features
elif args.arch == 'vgg16':
    print(f"Using {args.arch} as a pretrained model")
    model = models.vgg16_bn(pretrained=True)
    input_size = model.classifier[0].in_features
else:
    print(f"{args.arch} is not available. Kindly model available from this list only as a pretrained model. Otherwise an error is returned and no training will be done")
    
    
# Use GPU if it's available
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
 
hidden_units = args.hidden_units
output_size = 102
model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, output_size),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

# Training
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()



# Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_units': hidden_units,
              'architecture' : args.arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': epochs}

torch.save(checkpoint, 'checkpoint.pth')
