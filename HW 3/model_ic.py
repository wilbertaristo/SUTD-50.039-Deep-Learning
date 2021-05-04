import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

# Define classifier class
class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

""" ========= CUSTOM MODELS THAT I MADE (WILBERT ARISTO) ======== """
class TwoLayerConvNet(nn.Module):
  def __init__(self):
    super(TwoLayerConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 12, 5, 1, 2)
    self.conv2 = nn.Conv2d(12, 20, 5, 1, 2)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(56 * 56 * 20, 120)
    self.classifier = nn.Linear(120, 10)

  def forward(self, x):
    # Input Size: [64, 3, 224, 224]
    # Go through first convolutional layer & pooling
    x = self.pool(F.relu(self.conv1(x)))  # -> [64, 12, 112, 112]
    # Go through second convolutional layer & pooling
    x = self.pool(F.relu(self.conv2(x)))  # -> [64, 20, 56, 56]
    # Flatten
    x = x.view(-1, 20 * 56 * 56)          # -> [64, 62720]
    # Go through second last fully-connected layer
    x = F.relu(self.fc1(x))               # -> [64, 120]
    # Go through last fully-connected layer
    x = self.classifier(x)                # -> [64, 10]
    return x
  
class FiveLayerConvNet(nn.Module):
  def __init__(self):
    super(FiveLayerConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 12, 3, 1, 1)
    self.conv2 = nn.Conv2d(12, 20, 3, 1, 1)
    self.conv3 = nn.Conv2d(20, 32, 3, 1, 1)
    self.conv4 = nn.Conv2d(32, 48, 3, 1, 1)
    self.conv5 = nn.Conv2d(48, 68, 3, 1, 1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(7 * 7 * 68, 120)
    self.classifier = nn.Linear(120, 10)

  def forward(self, x):
    # Input Size: [64, 3, 224, 224]
    # Go through first convolutional layer & pooling
    x = self.pool(F.relu(self.conv1(x)))  # -> [64, 12, 112, 112]
    # Go through second convolutional layer & pooling
    x = self.pool(F.relu(self.conv2(x)))  # -> [64, 20, 56, 56]
    # Go through third convolutional layer & pooling
    x = self.pool(F.relu(self.conv3(x)))  # -> [64, 32, 28, 28]
    # Go through fourth convolutional layer & pooling
    x = self.pool(F.relu(self.conv4(x)))  # -> [64, 48, 14, 14]
    # Go through fifth convolutional layer & pooling
    x = self.pool(F.relu(self.conv5(x)))  # -> [64, 60, 7, 7]
    # Flatten
    x = x.view(-1, 68 * 7 * 7)            # -> [64, 3332]
    # Go through first fully-connected layer
    x = F.relu(self.fc1(x))               # -> [64, 120]
    # Go through last fully-connected layer
    x = self.classifier(x)                # -> [64, 10]
    return x

""" ============ END OF CUSTOM MODELS THAT I MADE (WILBERT ARISTO) ======== """

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define NN function
""" (WILBERT ARISTO)
    I ADDED 3 EXTRA ARGUMENTS INTO make_NN function namely the arguments:
                pretrain, finetune_whole, custom_model
    1) pretrain indicates whether the model is previously pretrained by other datasets or not
    2) finetune_whole indicates whether we should freeze the parameters & optimize the whole model paramaters in the Adam optimizer
    3) custom_model indicates which custom CNN to use for model (2 layer or 5 layer CNN that I created above)
    """
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, pretrain, finetune_whole, custom_model):
    """ IMPLEMENTATION OF custom_model ARGUMENT TO DECIDE WHICH MODEL TO USE (WILBERT ARISTO) """
    if custom_model == 2:
      # Use custom two-layer convolution model
      print("Using Two-Layer CNN")
      model = TwoLayerConvNet()
    elif custom_model == 5:
      print("Using Five-Layer CNN")
      # Use custom five-layer convolution model
      model = FiveLayerConvNet()
    else:
      # Import NN model (either pretrained or not)
      model = getattr(models, model_name)(pretrained=pretrain)
    """ ===================================================================================== """"
    
    """ IMPLEMENTATION OF finetune_whole ARGUMENT TO EITHER FREEZE THE PARAMETERS OR NOT (WILBERT ARISTO) """
    # If we do not need to finetune whole model, freeze parameters that we don't need to re-train
    if not finetune_whole:
      for param in model.parameters():
          param.requires_grad = False
    """ ===================================================================================== """"

    n_out = len(labelsdict)

    """ CHANGED LAST LAYER TO model.fc IF WE ARE USING RESNET MODEL (WILBERT ARISTO) """
    if "resnet" in model_name:
      # Make classifier
      n_in = next(model.fc.modules()).in_features
      model.fc = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
      
      """ IMPLEMENTATION OF finetune_whole ARGUMENT TO EITHER OPTIMIZE ALL PARAMETERS OR JUST THE LAST LAYER'S PARAMS (WILBERT ARISTO) """
      # Define optimizer
      if finetune_whole:
        optimizer = optim.Adam(model.parameters(), lr = lr)
      else:
        optimizer = optim.Adam(model.fc.parameters(), lr = lr)
      """ ============================================================================================================================ """"
    else:
      # Make classifier
      n_in = next(model.classifier.modules()).in_features
      model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
      
      """ IMPLEMENTATION OF finetune_whole ARGUMENT TO EITHER OPTIMIZE ALL PARAMETERS OR JUST THE LAST LAYER'S PARAMS (WILBERT ARISTO) """
      # Define optimizer
      if finetune_whole:
        optimizer = optim.Adam(model.parameters(), lr = lr)
      else:
        optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
      """ ============================================================================================================================ """"
    """ ============================================================================================================================ """"

    # Define criterion
    criterion = nn.NLLLoss()  

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0 
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
    
    """ CHANGED LAST LAYER TO model.fc IF WE ARE USING RESNET MODEL (WILBERT ARISTO) """
    if "resnet" in model_name:
      # Add model info 
      model.fc.n_in = n_in
      model.fc.n_hidden = n_hidden
      model.fc.n_out = n_out
      model.fc.labelsdict = labelsdict
      model.fc.lr = lr
      model.fc.optimizer_state_dict = optimizer.state_dict
      model.fc.model_name = model_name
      model.fc.class_to_idx = train_data.class_to_idx
    else:
      # Add model info 
      model.classifier.n_in = n_in
      model.classifier.n_hidden = n_hidden
      model.classifier.n_out = n_out
      model.classifier.labelsdict = labelsdict
      model.classifier.lr = lr
      model.classifier.optimizer_state_dict = optimizer.state_dict
      model.classifier.model_name = model_name
      model.classifier.class_to_idx = train_data.class_to_idx
    """ ============================================================================================================================ """"

    print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model

# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    
# Define function to load model
def load_model(path):
    cp = torch.load(path)
    
    # Import pre-trained NN model 
    model = getattr(models, cp['model_name'])(pretrained=True)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False
    
    # Make classifier
    model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                     hidden_layers=cp['c_hidden'])
    
    # Add model info 
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    
    return model

def test_model(model, testloader, device='cuda'):  
    model.to(device)
    model.eval()
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))