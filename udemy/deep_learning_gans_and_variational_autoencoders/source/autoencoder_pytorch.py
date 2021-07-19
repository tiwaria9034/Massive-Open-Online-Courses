"""
Code to implement Auto-encoder using PyTorch.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import utils.io_utils as io_utils


class Autoencoder(nn.Module):
    def __init__(self, num_feature_dim, num_hidden_layers):
        super(Autoencoder, self).__init__()

        # Encoder & Decoder
        self.enc = nn.Linear(in_features=num_feature_dim, out_features=num_hidden_layers)
        self.dec = nn.Linear(in_features=num_hidden_layers, out_features=num_feature_dim)

    def forward(self, x):
        """Forward pass given data point."""
        x = F.relu(self.enc(x))
        x = F.relu(self.dec(x))
        return x


def train(data, num_epochs=30, batch_size=128):
    """Perform model training."""
    # Create model
    model = Autoencoder(num_feature_dim=784, num_hidden_layers=300)

    # Specify loss and optimizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to target device
    model.to(device)

    # Perform training
    loss_values = []
    num_batches = len(data) // batch_size
    for epoch in tqdm(range(num_epochs), desc="Training Epoch"):
        # Shuffle input data
        np.random.shuffle(data)
        running_loss = 0.0

        # Update for each batch data
        for batch_idx in range(num_batches):
            batch_data = torch.Tensor(data[batch_idx * batch_size:(batch_idx + 1) * batch_size])
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_values.append(running_loss / len(data))

    plt.plot(loss_values)
    plt.grid()
    plt.title("Training Loss vs. Epochs")
    plt.show()

    return model, device


def main():
    """Main driver method."""
    # Fit classifier to MNIST data
    data, _ = io_utils.load_mnist(data_dir=Path(BASE_DIR, "data", "digit-recognizer"),
                                  mode="train")

    # Get trained model
    model, device = train(data=data,
                          num_epochs=50,
                          batch_size=512)

    # Plot reconstruction
    num_samples = 10
    while num_samples > 0:
        i = np.random.choice(len(data))
        data_instance = torch.Tensor(data[i]).to(device)
        im = model(data_instance).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(data_instance.cpu().detach().numpy().reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(im.cpu().detach().numpy(), cmap="gray")
        plt.title("Reconstruction")
        plt.show()

        num_samples -= 1


if __name__ == "__main__":
    main()
