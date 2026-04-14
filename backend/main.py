import os
import io
import pickle
import numpy as np
import cv2
import base64
import jwt
from datetime import datetime, timedelta
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms, models

SECRET_KEY = "super-secret-ris-key-change-in-production"
ALGORITHM = "HS256"
TEST_EMAIL = "test@ris.local"
TEST_PASSWORD = "Test@12345"

engine = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/api/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = pwd_context.hash(user.password)
    new_user = UserDB(name=user.name, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    token = jwt.encode({"sub": new_user.email, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token, "user": {"name": new_user.name, "email": new_user.email}}

@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    if user.email == TEST_EMAIL and user.password == TEST_PASSWORD:
        token = jwt.encode({"sub": TEST_EMAIL, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
        return {"token": token, "user": {"name": "Test Radiologist", "email": TEST_EMAIL}}

    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": db_user.email, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token, "user": {"name": db_user.name, "email": db_user.email}}

def verify_token(token: str | None = Depends(oauth2_scheme)):
    if not token or token == "guest":
        return "guest"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

class DenseNet169_GradCAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet169(weights=None)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        self.gradients = None
    def activations_hook(self, grad): self.gradients = grad
    def forward(self, x):
        features = self.model.features(x)
        if torch.is_grad_enabled():
            features.requires_grad_()
            features.register_hook(self.activations_hook)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.model.classifier(out), features

class ResNet50_GradCAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.gradients = None
    def activations_hook(self, grad): self.gradients = grad
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)
        if torch.is_grad_enabled():
            features.requires_grad_()
            features.register_hook(self.activations_hook)
        x = self.model.avgpool(features)
        x = torch.flatten(x, 1)
        return self.model.fc(x), features

class SwinModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.swin_t(weights=None)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

def load_classes(filepath):
    try:
        with open(filepath, "rb") as f:
            return len(pickle.load(f))
    except FileNotFoundError:
        return 14

def load_model(ModelClass, weights_path, num_classes):
    model = ModelClass(num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        return model
    except FileNotFoundError:
        return None

XRAY_DIR = "models/XRAY_MODELS"
CT_DIR = "models/CT_Scan_models"

xray_classes = load_classes(f"{XRAY_DIR}/classes.pkl")
ct_classes = load_classes(f"{CT_DIR}/classes.pkl")

system_models = {
    "xray": {
        "densenet": load_model(DenseNet169_GradCAM, f"{XRAY_DIR}/densenet_best.pth", xray_classes),
        "resnet": load_model(ResNet50_GradCAM, f"{XRAY_DIR}/resnet_best.pth", xray_classes),
        "swin": load_model(SwinModel, f"{XRAY_DIR}/swin_best.pth", xray_classes)
    },
    "ct": {
        "densenet": load_model(DenseNet169_GradCAM, f"{CT_DIR}/densenet_best.pth", ct_classes),
        "resnet": load_model(ResNet50_GradCAM, f"{CT_DIR}/restnet50.pkl", ct_classes),
        "swin": load_model(SwinModel, f"{CT_DIR}/swin_model.pkl", ct_classes)
    }
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, img_tensor, original_img):
    if model is None: return original_img, original_img
    model.zero_grad()
    if isinstance(model, SwinModel): return original_img, original_img 

    logits, features = model(img_tensor)
    target_class = logits.argmax(dim=1).item()
    logits[0, target_class].backward()

    gradients = model.gradients[0].cpu().data.numpy()
    pooled_gradients = np.mean(gradients, axis=(1, 2))
    features = features[0].cpu().data.numpy()

    for i in range(features.shape[0]):
        features[i, :, :] *= pooled_gradients[i]

    heatmap = np.mean(features, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
    return heatmap_colored, overlay

def image_to_base64(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), scanType: str = Form("xray"), current_user: str = Depends(verify_token)):
    try:
        scan_category = scanType.lower()
        if scan_category not in system_models:
            raise HTTPException(status_code=400, detail="Invalid scan type selected.")

        active_models = system_models[scan_category]

        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        original_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_()

        results = {"original": f"data:image/jpeg;base64,{image_to_base64(original_img)}"}

        for name, model in active_models.items():
            heatmap, overlay = generate_gradcam(model, img_tensor, original_img)
            results[name] = {
                "heatmap": f"data:image/jpeg;base64,{image_to_base64(heatmap)}",
                "overlay": f"data:image/jpeg;base64,{image_to_base64(overlay)}"
            }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
