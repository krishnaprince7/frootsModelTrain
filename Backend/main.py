from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# 🔥 Tumhari nayi 15 classes (Folders ke exact naam)
classes = [
    'AppleFormalin-mixed', 'AppleFresh', 'AppleRotten',
    'BananaFormalin-mixed', 'BananaFresh', 'BananaRotten',
    'GrapeFormalin-mixed', 'GrapeFresh', 'GrapeRotten',
    'MangoFormalin-mixed', 'MangoFresh', 'MangoRotten',
    'OrangeFormalin-mixed', 'OrangeFresh', 'OrangeRotten'
]

print("Naya Super AI load ho raha hai, wait karo...")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

try:
    # Ab naya model load ho raha hai
    model.load_state_dict(torch.load('all_crops_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ Super AI Brain successfully FastAPI me lag gaya hai!")
except Exception as e:
    print("⚠️ Error:", e)

# Image preprocessing rules
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "AgriSmart AI (Multi-Crop Super Edition) ka Backend chal raha hai! 🚀"}

@app.post("/agridata")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        tensor = transform(image).unsqueeze(0) 
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        result = classes[predicted.item()]
        
        return {
            "status": "success",
            "disease": result,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}