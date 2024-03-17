import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TwoLayerNet
from data_dnLoader import mnist

# w = torch.rand(7, 7)
# U, S, V = torch.svd(w, some=False)
# print(w-U@torch.diag(S)@V.t())

# exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images , train_labels = mnist( train=True )

input_size = 784 
hidden_size = 128
output_size = 10 

model = TwoLayerNet(input_size, hidden_size, output_size).to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(param.shape)

print_trainable_parameters(model)

num_epochs = 2
for epoch in range(num_epochs):

    for i,image in enumerate(train_images):
        optimizer.zero_grad()
        
        label = torch.zeros(10)
        label[ train_labels[i] ] = 1

        outputs = model(torch.tensor(image,device=device).flatten())

        loss = criterion(outputs, torch.tensor(label,device=device).flatten())

        loss.backward()
        optimizer.step()


        if i & 4095 == 0 : print(loss)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        

test_images, test_labels = mnist(train=False)

model.eval()  # Set the model to evaluation mode

total = 0
correct = 0

# No need to compute gradients during evaluation/testing
with torch.no_grad():
    for i, image in enumerate(test_images):
        label = test_labels[i]

        outputs = model(torch.tensor(image, device=device).flatten())
        # Get predictions from the maximum value
        predicted = torch.argmax(outputs)

        # Total number of labels
        total += 1

        # Total correct predictions
        if predicted == torch.tensor(label, device=device):
            correct += 1

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy} %')

