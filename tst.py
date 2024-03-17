import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TwoLayerNet
from data_dnLoader import mnist

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

input_size, hidden_size, output_size = 784 , 784 , 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoLayerNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( 
    model.parameters(),
    lr=1.0/(1<<8), 
    eps=1.0/(1<<26),
)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

train_images , train_labels = mnist( train=True )

model.train()
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

correct = 0
with torch.no_grad():
    for i, image in enumerate(test_images):
        label = test_labels[i]

        outputs = model(torch.tensor(image, device=device).flatten())
        predicted = torch.argmax(outputs)

        correct += predicted==label

print(f'Acc: { 100 * correct/len(test_images) } %')
