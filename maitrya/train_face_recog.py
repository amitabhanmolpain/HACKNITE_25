import kagglehub
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision

# Download FER2013 dataset
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

# Define EmotionModel using ResNet18
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)  # Adjust for 7 emotions

    def forward(self, x):
        return self.model(x)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted for 3 channels
])

# Load dataset from downloaded path
train_dataset = datasets.ImageFolder(f"{path}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, optimizer
model = EmotionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # Set learning rate
scaler = GradScaler()  # Mixed Precision Training

num_epochs = 10  # Adjust as needed

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "emotion_model.pth")
print("Model saved successfully!")
