import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import os
import urllib.request
import zipfile
import tarfile

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#parameters for training
batch_size = 64
learning_rate = 0.001
num_epochs = 5

#only needed to change if using a custom dataset, otherwise defaults to MNIST (grayscale 28x28)
IMAGE_CHANNELS = 3
IMAGE_SIZE = 32

#place custom dataset here
#example dataset included below:
CUSTOM_DATASET_URL = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
DATASET_NAME = "Sign-Language-Digits-Dataset-master"  
def download_and_extract(url: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    filename = os.path.join(dest, os.path.basename(url))
    print(f"Downloading {url} -> {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as z:
            z.extractall(dest)
    elif filename.endswith('.tar.gz') or filename.endswith('.tgz') or filename.endswith('.tar'):
        with tarfile.open(filename, 'r:*') as t:
            t.extractall(dest)
    else:
        print("Can't download file. Double check url is right.")


# dataset generation
if CUSTOM_DATASET_URL:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    

    download_and_extract(CUSTOM_DATASET_URL, './data/' + DATASET_NAME)

    root_candidates = [os.path.join('./data/' + DATASET_NAME, d) for d in os.listdir('./data/' + DATASET_NAME)]
    dataset_root = './data/' + DATASET_NAME
    for c in root_candidates:
        if os.path.isdir(c) and any(os.path.isdir(os.path.join(c, x)) for x in os.listdir(c)):
            dataset_root = c
            break

    full_dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_len, test_len])

else:
    #default MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    IMAGE_CHANNELS = 1
    IMAGE_SIZE = 28

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#different models that were implemented for the MNIST dataset, starting with a simple logistic regression, then a multi-layer perceptron (MLP), and finally a convolutional neural network (CNN). 
# Each model is defined as a subclass of `nn.Module` and implements the `forward` method to specify how the input data flows through the network.
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


#how to train and evaluate the models. 
#model gets trained using loss function and optimizer
def train_model(model):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return losses



# iterates over the test data, and computes the accuracy by comparing the predicted labels with the true labels. 
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            unused, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Run the models
models = {
    "Logistic Regression": LogisticRegression(),
    "MLP": MLP(),
    "CNN": CNN()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}")
    losses = train_model(model)
    accuracy, preds, labels = evaluate_model(model)
    params = count_parameters(model)

    results[name] = {
        "accuracy": accuracy,
        "params": params,
        "losses": losses,
        "preds": preds,
        "labels": labels
    }

    print(f"{name} Accuracy: {accuracy:.2f}%")
    print(f"{name} Parameters: {params:,}")


#Plot training loss curves
plt.figure()
for name in results:
    plt.plot(results[name]["losses"], label=name)

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


#Plot Confusion Matrix
for name in results:
    preds = results[name]["preds"]
    labels = results[name]["labels"]
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure()
    sns.heatmap(cm, annot=False, fmt="d")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


#Print a final summary in a clear and concise way
print("\n=== Final Results ===")
for name in results:
    print(f"{name:20s} | "
          f"Accuracy: {results[name]['accuracy']:.2f}% | "
          f"Params: {results[name]['params']:,}")
