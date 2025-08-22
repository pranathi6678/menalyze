import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from classifier import SimpleCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

def evaluate_model(data_dir=DATA_DIR, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Evaluation complete: Accuracy = {acc:.2f}% on {total} samples")

if __name__ == "__main__":
    evaluate_model()
