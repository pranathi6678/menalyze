import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from classifier import SimpleCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def train_real_model(data_dir=DATA_DIR, epochs=5, batch_size=16, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    save_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model trained and saved to {save_path}")

if __name__ == "__main__":
    train_real_model()
