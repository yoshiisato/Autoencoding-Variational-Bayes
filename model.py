import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data.dataloader as Dataloader

import torch.optim as optim

# Hyperparameters
EPOCH = 15
lr = .0005
val_iter = 100
# ---------------

training_data = datasets.MNIST(
    root="data",
    train=True,
    tranform=ToTensor(),
    download=True,
)

testing_data = datasets.MNIST(
    root="data",
    train=False,
    tranform=ToTensor(),
    download=True,
)

train_dataloader = Dataloader(training_data, batch_size=64, shuffle=True)
test_dataloader = Dataloader(testing_data, batch_size=64, shuffle=False)


# implementation of the variational autoencoder from Kingma et al.
class VAE(nn.module):
    def __init__(self, latent_dim):
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=16)
        self.conv2 = nn.Conv2d(8, 16, 8)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.fc1 = nn.Linear(4*4*32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, latent_dim)
        self.fc3_stdv = nn.Linear(64, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 4*4*32)
        self.convt4 = nn.ConvTranspose2d(32, 16, 4)
        self.convt5 = nn.ConvTranspose2d(16, 8, 8)
        self.convt6 = nn.ConvTranspose2d(8, 1, 28)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        mean_input_vector = x
        stdv_input_vector = x
        mean_vector = self.fc3_mean(F.relu(mean_input_vector))
        stdv_vector = self.fc3_stdv(F.relu(stdv_input_vector))
        epsilon = torch.normal(mean=torch.zeros(8), std=torch.ones(8))

        # reparametrization trick
        reparam_vector = mean_vector + stdv_vector * epsilon

        # Decoder
        x = F.relu(self.fc4(reparam_vector))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = torch.unflatten(x, dim=0, sizes=(4, 4, 32))
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt5(x))
        x = F.relu(self.convt6(x))
        return x


# optimizer
model = VAE(latent_dim=8)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def vae_loss(outputs, labels):
    return 0


def val_check(inputs, labels):
    datalaoder_len = len(test_dataloader)
    for i, batch in enumerate(test_dataloader):
        running_loss = 0.0
        outputs = model(inputs)
        loss = vae_loss(outputs, labels)
        running_loss += loss.item()

        return running_loss


# Training Loop
for epoch in EPOCH:
    running_loss = 0.0
    total_iter = 0

    for i, batch in enumerate(train_dataloader):
        total_iter += 1
        inputs, labels = batch
        labels = inputs
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = vae_loss(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if total_iter % val_iter == 0:
            

