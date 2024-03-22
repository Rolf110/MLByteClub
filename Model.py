import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import copy 
import numpy as np
import os


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_1=100, hidden_size_2=100):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=5, padding=1)

        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 64)
        self.linear3 = nn.Linear(64, output_size)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        x = self.linear3(x)
        return x


def train(model, dataloader, val_dataloader, epochs, patience, device='cpu', verbose = True):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    early_stopping_counter = 0
    min_val_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(epochs)):
        total_loss = 0

        model.train()
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = loss_fn(output, target)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data, val_target in val_dataloader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                v_loss = loss_fn(val_output, val_target)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        if verbose:
            print(f'Epoch: {epoch+1} \tTraining Loss: {avg_loss:.4f} \tValidation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping")
            break

    if best_model:
        print("Loading best model weights!")
        model.load_state_dict(best_model)

    return model


def test(model, dataloader, device='cpu'):
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            _, predicted = torch.max(output.data, 1)

            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

    accuracy = 100 * correct_predictions / total_predictions
    return np.round(accuracy, 2)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')
