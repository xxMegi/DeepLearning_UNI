import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.05
SEED = 42
DATA_DIR = "data"
OUTPUT_DIR = Path("wyniki")


class HighwayLayer(nn.Module):
    """Jedna warstwa Highway: y = H(x)*T(x) + x*(1-T(x))"""

    def __init__(self, channels):
        super().__init__()

        # H(x): zwykle przeksztalcenie obrazu
        self.transform = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # T(x): bramka, ktora wybiera udzial H(x) i oryginalnego x
        self.gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Na poczatku siec ma chetniej przepuszczac oryginalne informacje
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x):
        h = torch.relu(self.transform(x))
        t = torch.sigmoid(self.gate(x))
        y = h * t + x * (1 - t)
        return y


class HighwayNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Normalne warstwy moga zmieniac liczbe kanalow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Highway zachowuje liczbe kanalow i rozmiar obrazu
        self.highway1 = HighwayLayer(32)
        self.highway2 = HighwayLayer(32)
        self.highway3 = HighwayLayer(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.highway4 = HighwayLayer(64)
        self.highway5 = HighwayLayer(64)
        self.highway6 = HighwayLayer(64)

        # Srednia wartosc w kazdym kanale
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # [B, 32, 32, 32]
        x = self.pool(x)                # [B, 32, 16, 16]
        x = self.highway1(x)
        x = self.highway2(x)
        x = self.highway3(x)

        x = torch.relu(self.conv2(x))   # [B, 64, 16, 16]
        x = self.pool(x)                # [B, 64, 8, 8]
        x = self.highway4(x)
        x = self.highway5(x)
        x = self.highway6(x)

        x = self.average_pool(x)        # [B, 64, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [B, 64]
        x = self.classifier(x)          # [B, 10]
        return x


def load_data():
    # Podstawowe przetwarzanie pikseli: tensor i normalizacja
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                     transform=train_transform)
    validation_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                          transform=test_transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                    transform=test_transform)

    indices = list(range(len(train_dataset)))
    random.Random(SEED).shuffle(indices)
    train_indices = indices[:45000]
    validation_indices = indices[45000:]

    train_loader = DataLoader(Subset(train_dataset, train_indices),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(Subset(validation_dataset, validation_indices),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


def save_results(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "historia.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="Trening")
    plt.plot(epochs, val_loss, label="Walidacja")
    plt.xlabel("Epoka")
    plt.ylabel("Loss (cross-entropy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_acc, label="Trening")
    plt.plot(epochs, val_acc, label="Walidacja")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png")
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Urzadzenie:", device)

    train_loader, validation_loader, test_loader = load_data()
    model = HighwayNetwork().to(device)
    print("Liczba parametrow:", sum(p.numel() for p in model.parameters()))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = evaluate(
            model, validation_loader, loss_function, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(f"Epoka {epoch:02d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f}, acc: {train_accuracy:.2f}% | "
              f"val loss: {val_loss:.4f}, acc: {val_accuracy:.2f}%")

        # Model z najlepszym wynikiem walidacyjnym
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), OUTPUT_DIR / "highway_model.pth")

        scheduler.step()
        save_results(history)

    # Test po wybraniu najlepszego modelu
    model.load_state_dict(torch.load(OUTPUT_DIR / "highway_model.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    print(f"\nWYNIK TESTOWY: loss = {test_loss:.4f}, accuracy = {test_accuracy:.2f}%")

    with open(OUTPUT_DIR / "wynik_testowy.txt", "w", encoding="utf-8") as file:
        file.write(f"Test loss: {test_loss:.4f}\n")
        file.write(f"Test accuracy: {test_accuracy:.2f}%\n")
        file.write(f"Best validation accuracy: {best_val_accuracy:.2f}%\n")


if __name__ == "__main__":
    main()