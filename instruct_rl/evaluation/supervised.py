import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import LevelDataset

# 1. data load
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

input_images = dataset.images #  3664, 16, 16, 3
labels = dataset.dataframe['type'].values


print(f"Dataset shape: {input_images.shape}")
print(f"Number of labels: {len(labels)}")

# 2. define Custom Dataset
class LevelTorchDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.label_set = list(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (16, 16, 3)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, 16, 16)
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        return image, label_idx

# 3. change CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # convert input channel 3
            nn.MaxPool2d(2),  # 16x16 → 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 → 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 4. prepare Dataset, DataLoader
torch_dataset = LevelTorchDataset(input_images, labels)
dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

# 5. setting model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(torch_dataset.label_set)

model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. train loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, label_indices in dataloader:
        images, label_indices = images.to(device), label_indices.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, label_indices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 7. save model
torch.save(model.state_dict(), "trained_level_model.pth")

