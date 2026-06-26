import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


class SimCLRTransform:

    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class SimCLRDataset(Dataset):


    def __init__(self,
                 root="../data",
                 train=True,
                 download=True):

        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=download
        )

        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, _ = self.dataset[index]

        view1, view2 = self.transform(image)

        return view1, view2