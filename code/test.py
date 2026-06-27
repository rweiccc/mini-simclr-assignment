import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SimCLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])


test_dataset = CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


simclr = SimCLR()

simclr.load_state_dict(
    torch.load(
        "checkpoints/simclr.pth",
        map_location=DEVICE
    )
)

simclr.to(DEVICE)
simclr.eval()

classifier = nn.Linear(512, 10).to(DEVICE)

classifier.load_state_dict(
    torch.load(
        "checkpoints/linear_probe.pth",
        map_location=DEVICE
    )
)

classifier.to(DEVICE)
classifier.eval()

correct = 0
total = 0

show_images = []
show_preds = []
show_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = simclr.encode(images)
        outputs = classifier(features)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # collect first 5 images
        if len(show_images) < 5:
            remain = 5 - len(show_images)

            show_images.extend(images[:remain].cpu())
            show_preds.extend(predicted[:remain].cpu())
            show_labels.extend(labels[:remain].cpu())


accuracy = 100.0 * correct / total


print("=" * 50)
print(f"Test Accuracy: {accuracy:.2f}%")
print("=" * 50)

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

plt.figure(figsize=(15,3))

for i in range(5):

    img = show_images[i]

    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    plt.subplot(1, 5, i+1)
    plt.imshow(img.permute(1,2,0))
    plt.axis("off")

    plt.title(
        f"P:{classes[show_preds[i]]}\nT:{classes[show_labels[i]]}",
        fontsize=9
    )

plt.tight_layout()

plt.savefig("results/test_predictions.png")

plt.show()


print("Prediction image saved to:")
print("results/test_predictions.png")