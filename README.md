import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cpu")
print("Using device:", device)

# ---------------------------
# TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------------
# LOAD DATA
# ---------------------------
train_full = datasets.ImageFolder("fer2013/train", transform=transform)
test_full  = datasets.ImageFolder("fer2013/test", transform=transform)

# use small subset for demo (FAST)
train_data = Subset(train_full, range(3000))
test_data  = Subset(test_full, range(1000))

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print("Train samples:", len(train_data))
print("Test samples:", len(test_data))

# ---------------------------
# MODEL
# ---------------------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = EmotionCNN().to(device)


# LOSS & OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING (FAST)

epochs = 1

for epoch in range(epochs):
    print("Starting Epoch", epoch + 1)
    model.train()

    or i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        ofptimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("Batch", i, "Loss:", round(loss.item(), 4))

        if i == 5:   # stop early for demo
            break

    print("Epoch finished\n")


# TESTING

model.eval()
correct = 0
total = 0
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print("Test Accuracy:", round(accuracy, 2), "%")


# CONFUSION MATRIX

cm = confusion_matrix(true_labels, pred_labels)

plt.imshow(cm)
plt.title("FER2013 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


# SAVE MODEL

torch.save(model.state_dict(), "emotion_model_demo.pth")
print("Model saved as emotion_model_demo.pth")
