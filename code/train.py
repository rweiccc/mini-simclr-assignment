import os
import torch
from torch.utils.data import DataLoader

from dataset import SimCLRDataset
from model import SimCLR
from loss import NTXentLoss


BATCH_SIZE = 64
EPOCHS = 2
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_dataset = SimCLRDataset(
        root="data",
        train=True,
        download=False   
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

  
    model = SimCLR(
        feature_dim=128,
        pretrained=False
    ).to(DEVICE)


    criterion = NTXentLoss(temperature=0.5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    loss_log = open("logs/loss.txt", "w")

    print("=" * 50)
    print("Start Training")
    print("=" * 50)

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0.0

        for step, (x1, x2) in enumerate(train_loader):

            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        loss_log.write(f"{epoch+1},{avg_loss:.6f}\n")

    loss_log.close()

    torch.save(model.state_dict(), "checkpoints/simclr.pth")

    print("=" * 50)
    print("Training Finished")
    print("Model Saved to checkpoints/simclr.pth")
    print("=" * 50)


if __name__ == "__main__":
    main()