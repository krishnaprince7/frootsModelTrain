import torch
from torchvision import datasets, transforms

# Naya Path
data_dir = './All_Crops_Dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"✅ Total Images: {len(dataset)}")
print(f"🔥 Total Classes: {len(dataset.classes)}")
print(f"👇 Ye list copy kar lo (main.py ke liye):\n{dataset.classes}")