import kagglehub
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# Download FER2013 dataset from Kaggle
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
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset from downloaded path
train_dataset = datasets.ImageFolder(f"{path}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set training parameters
num_epochs = 10  # Adjust this value as needed
model = EmotionModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "emotion_model.pth")
print("Model saved successfully!")
