import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from model import SimCLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

os.makedirs("../checkpoints", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

train_dataset = CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

simclr = SimCLR()

simclr.load_state_dict(
    torch.load(
        "../checkpoints/simclr.pth",
        map_location=DEVICE
    )
)

simclr.to(DEVICE)
simclr.eval()

for param in simclr.parameters():
    param.requires_grad = False


classifier = nn.Linear(512, 10).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=LR
)

print("=" * 50)
print("Start Linear Probe Training")
print("=" * 50)

for epoch in range(EPOCHS):

    classifier.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            features = simclr.encode(images)

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

classifier.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = simclr.encode(images)

        outputs = classifier(features)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total

print("=" * 50)
print(f"Linear Probe Accuracy: {accuracy:.2f}%")
print("=" * 50)

torch.save(
    classifier.state_dict(),
    "../checkpoints/linear_probe.pth"
)

print("Linear Probe Model Saved!")