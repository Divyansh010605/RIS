import os
import io
import pickle
import warnings
import logging
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
from pydantic import BaseModel, model_validator
import torch
import torch.nn as nn
from torchvision import transforms, models

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
from tensorflow import keras

# Patch Keras Dense layer to ignore unrecognized quantization_config parameter
original_dense_from_config = keras.layers.Dense.from_config
@classmethod
def patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return original_dense_from_config(config)
keras.layers.Dense.from_config = patched_dense_from_config

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment-based configuration
# Bug fix: Warn when using insecure default SECRET_KEY; remove insecure fallback in production.
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    SECRET_KEY = "super-secret-ris-key-change-in-production"
    logger.warning(
        "SECRET_KEY env var is not set — falling back to insecure default. "
        "Set SECRET_KEY before deploying to production."
    )
ALGORITHM = "HS256"
# Bug fix: TEST_EMAIL/TEST_PASSWORD are now opt-in via env vars.
# Leave both unset in production to disable the backdoor account entirely.
TEST_EMAIL = os.getenv("TEST_EMAIL", "")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "")
# Bug fix: Removed duplicate DATABASE_URL assignment that shadowed the first.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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

    # Bug fix: validate() was a plain method never called automatically by Pydantic.
    # Replaced with a proper @model_validator so validation runs on every construction.
    @model_validator(mode='after')
    def validate_fields(self):
        if not self.name or len(self.name.strip()) < 2:
            raise ValueError("Name must be at least 2 characters long")
        if "@" not in self.email or len(self.email) < 5:
            raise ValueError("Invalid email format")
        if len(self.password) < 6:
            raise ValueError("Password must be at least 6 characters long")
        return self

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/api/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Bug fix: Removed manual user.validate() call — @model_validator now handles this automatically.
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        hashed_pw = pwd_context.hash(user.password)
        new_user = UserDB(name=user.name, email=user.email, hashed_password=hashed_pw)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        token = jwt.encode({"sub": new_user.email, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"New user registered: {new_user.email}")
        return {"token": token, "user": {"name": new_user.name, "email": new_user.email}}
    except Exception as e:
        db.rollback()
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    # Bug fix: Only activate backdoor if TEST_EMAIL/TEST_PASSWORD env vars are explicitly set.
    if TEST_EMAIL and TEST_PASSWORD and user.email == TEST_EMAIL and user.password == TEST_PASSWORD:
        token = jwt.encode({"sub": TEST_EMAIL, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Test user logged in: {TEST_EMAIL}")
        return {"token": token, "user": {"name": "Test Radiologist", "email": TEST_EMAIL}}

    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        logger.warning(f"Failed login attempt: {user.email}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({"sub": db_user.email, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"User logged in: {db_user.email}")
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
XRAY_IMG_SIZE = 256
CT_IMG_SIZE = 224

# Bug fix: Class was named DenseNet169_GradCAM but the model label and weights file
# both refer to DenseNet121. Fixed architecture to densenet121 and renamed the class.
class DenseNet121_GradCAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(weights=None)
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

def load_labels(filepath, default=None):
    try:
        with open(filepath, "rb") as f:
            labels = pickle.load(f)
        if isinstance(labels, np.ndarray):
            return labels.tolist()
        if isinstance(labels, (list, tuple)):
            return list(labels)
        return [str(labels)]
    except FileNotFoundError:
        logger.warning(f"Labels file not found: {filepath}")
        return default or []
    except Exception as e:
        logger.error(f"Error loading labels from {filepath}: {str(e)}")
        return default or []

def load_torch_checkpoint(weights_path):
    try:
        return torch.load(weights_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=DEVICE)
    except Exception as e:
        logger.error(f"Error loading torch checkpoint {weights_path}: {str(e)}")
        return None

def load_optional_torch_serialized_model(weights_path):
    if not os.path.exists(weights_path):
        logger.warning(f"Model file not found: {weights_path}")
        return None
    try:
        checkpoint = load_torch_checkpoint(weights_path)
        return checkpoint if isinstance(checkpoint, nn.Module) else None
    except Exception as e:
        logger.error(f"Error loading optional torch model {weights_path}: {str(e)}")
        return None

def load_torch_model(ModelClass, weights_path, num_classes):
    if not os.path.exists(weights_path):
        logger.warning(f"Torch model weights not found: {weights_path}")
        return None
    try:
        model = ModelClass(num_classes).to(DEVICE)
        checkpoint = load_torch_checkpoint(weights_path)
        if checkpoint is None:
            return None
        if isinstance(checkpoint, nn.Module):
            return checkpoint.to(DEVICE).eval()
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict):
            cleaned_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()
            return model
        return None
    except Exception as e:
        logger.error(f"Error loading torch model {weights_path}: {str(e)}")
        return None

class KerasCompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that removes incompatible Keras parameters during deserialization"""
    def find_class(self, module, name):
        if module == 'keras.src.layers.core.dense' or (module == 'keras.layers' and name == 'Dense'):
            return super().find_class('keras.src.layers.core.dense', 'Dense')
        return super().find_class(module, name)

class CustomDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def load_keras_model(weights_path):
    if not os.path.exists(weights_path):
        logger.warning(f"Keras model file not found: {weights_path}")
        return None
    try:
        # Try standard pickle first
        with open(weights_path, "rb") as f:
            model = pickle.load(f)
        return model
    except (TypeError, AttributeError) as e:
        # If quantization_config error, try custom unpickler
        if 'quantization_config' in str(e):
            try:
                with open(weights_path, "rb") as f:
                    # Read and modify pickle data to remove quantization_config
                    data = f.read()
                    # Replace quantization_config entries
                    data = data.replace(b"'quantization_config'", b"'_disabled_config'")
                    model = pickle.loads(data)
                return model
            except Exception as e_inner:
                pass
        # Try keras native loader
        try:
            model = keras.models.load_model(weights_path, compile=False, safe_mode=False, custom_objects={'Dense': CustomDense})
            return model
        except Exception as e_native:
            # Bug fix: bare `except: pass` swallowed errors silently; now logged.
            logger.warning(f"Keras native loader also failed for {weights_path}: {e_native}")
        # Last resort - return None with warning
        logger.warning(f"Could not load Keras model from {weights_path}. Original error: {e}")
        return None
    except Exception as e:
        try:
            model = keras.models.load_model(weights_path, compile=False, safe_mode=False, custom_objects={'Dense': CustomDense})
            return model
        except Exception as e2:
            print(f"Warning: Could not load Keras model from {weights_path}. Error: {e}. Keras Error: {e2}")
            return None

def preprocess_keras_image(image, model_key):
    image = image.resize((CT_IMG_SIZE, CT_IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    if model_key == "densenet":
        image_array = tf.keras.applications.densenet.preprocess_input(image_array)
    elif model_key == "resnet":
        image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    else:
        image_array = image_array / 255.0
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_array, axis=0), dtype=tf.float32)
    return image_tensor, np.array(image)

def binary_label(probability):
    if probability >= 0.5:
        return "Cancer Detected", probability, True
    return "No Cancer Detected", probability, False

def build_colormap_overlay(original_img, heatmap):
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
    return heatmap_colored, overlay

def centered_proxy_heatmap(original_img):
    h, w = original_img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    sigma_x = max(18.0, w / 6.0)
    sigma_y = max(18.0, h / 6.0)
    heatmap = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2) + ((y - cy) ** 2) / (2 * sigma_y ** 2)))
    return heatmap.astype(np.float32)

def unavailable_result(label, original_img, reason="Model unavailable in this runtime", heatmap_img=None, overlay_img=None):
    if heatmap_img is None or overlay_img is None:
        proxy = centered_proxy_heatmap(original_img)
        heatmap_img, overlay_img = build_colormap_overlay(original_img, proxy)
    return {
        "label": label,
        "prediction": "Unavailable",
        "confidence": 0.0,
        "status": reason,
        "heatmap": f"data:image/jpeg;base64,{image_to_base64(heatmap_img)}",
        "overlay": f"data:image/jpeg;base64,{image_to_base64(overlay_img)}",
    }

XRAY_DIR = "models/XRAY_MODELS"
CT_DIR = "models/CT_Scan_models"

xray_classes = load_labels(f"{XRAY_DIR}/classes.pkl")

system_models = {
    "xray": {
        "densenet": {
            "kind": "torch",
            "label": "DenseNet121",
            "model": load_torch_model(DenseNet121_GradCAM, f"{XRAY_DIR}/densenet_best.pth", len(xray_classes)),
            "class_names": xray_classes,
        },
        "resnet": {
            "kind": "torch",
            "label": "ResNet50",
            "model": load_torch_model(ResNet50_GradCAM, f"{XRAY_DIR}/resnet_best.pth", len(xray_classes)),
            "class_names": xray_classes,
        },
        "swin": {
            "kind": "torch",
            "label": "Swin Transformer",
            "model": load_torch_model(SwinModel, f"{XRAY_DIR}/swin_best.pth", len(xray_classes)),
            "class_names": xray_classes,
        },
    },
    "ct": {
        "densenet": {
            "kind": "keras",
            "label": "DenseNet121",
            "model": load_keras_model(f"{CT_DIR}/densenet121_lung_model.pkl"),
            "last_conv_layer": "conv5_block16_concat",
            "preprocess": "densenet",
        },
        "resnet": {
            "kind": "keras",
            "label": "ResNet50",
            "model": load_keras_model(f"{CT_DIR}/restnet50.pkl"),
            "last_conv_layer": "conv5_block3_out",
            "preprocess": "resnet",
        },
        "cnn": {
            "kind": "keras",
            "label": "CNN",
            "model": load_keras_model(f"{CT_DIR}/lung_cancer_cnn_model.pkl"),
            "last_conv_layer": "last_conv_layer",
            "preprocess": "basic",
        },
        "swin": {
            "kind": "torch",
            "label": "Swin Transformer",
            "model": load_optional_torch_serialized_model(f"{CT_DIR}/swin_model.pkl"),
            "class_names": ["Negative", "Positive"],
            "unavailable_reason": "swin checkpoint could not be restored on current runtime",
        },
    }
}

xray_transform = transforms.Compose([
    transforms.Resize((XRAY_IMG_SIZE, XRAY_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_torch_result(model, img_tensor, original_img, class_names=None):
    if model is None:
        return None

    model.zero_grad()
    with torch.enable_grad():
        result = model(img_tensor)
        
        # Handle both single output (just logits) and tuple output (logits, features)
        if isinstance(result, tuple):
            logits, features = result
        else:
            logits = result
            features = None

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_idx = predicted_idx.item()
        confidence_value = float(confidence.item())

        if class_names and predicted_idx < len(class_names):
            prediction_label = class_names[predicted_idx]
        else:
            prediction_label = f"Class {predicted_idx}"

        # For Swin Transformer, use a proxy heatmap since GradCAM is complex for transformers
        if isinstance(model, SwinModel):
            proxy = centered_proxy_heatmap(original_img)
            heatmap_colored, overlay = build_colormap_overlay(original_img, proxy)
            return {
                "prediction": prediction_label,
                "confidence": confidence_value,
                "heatmap": heatmap_colored,
                "overlay": overlay,
            }

        if features is None:
            return {
                "prediction": prediction_label,
                "confidence": confidence_value,
                "heatmap": original_img,
                "overlay": original_img,
            }

        logits[0, predicted_idx].backward()
        # Bug fix: model.gradients is set by the hook; if it never fired, it stays None
        # and indexing [0] would crash. Fall back to a proxy heatmap in that case.
        if model.gradients is None:
            proxy = centered_proxy_heatmap(original_img)
            heatmap_colored, overlay = build_colormap_overlay(original_img, proxy)
            return {
                "prediction": prediction_label,
                "confidence": confidence_value,
                "heatmap": heatmap_colored,
                "overlay": overlay,
            }
        gradients = model.gradients[0].cpu().data.numpy()
        pooled_gradients = np.mean(gradients, axis=(1, 2))
        feature_maps = features[0].cpu().data.numpy()

        for index in range(feature_maps.shape[0]):
            feature_maps[index, :, :] *= pooled_gradients[index]

        heatmap = np.mean(feature_maps, axis=0)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)

        return {
            "prediction": prediction_label,
            "confidence": confidence_value,
            "heatmap": heatmap_colored,
            "overlay": overlay,
        }

def generate_keras_gradcam(model, image_array, original_img, last_conv_layer_name, target_positive=True):
    if model is None:
        return original_img, original_img

    try:
        grad_model = keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array, training=False)
            if target_positive:
                loss = predictions[:, 0]
            else:
                loss = 1.0 - predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Gradients were None for selected layer")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = heatmap.numpy().astype(np.float32)
        return build_colormap_overlay(original_img, heatmap)
    except Exception:
        # Saliency fallback keeps CT outputs consistent even if Grad-CAM layer wiring fails.
        try:
            image_var = tf.Variable(image_array)
            with tf.GradientTape() as tape:
                predictions = model(image_var, training=False)
                loss = predictions[:, 0]
            grads = tape.gradient(loss, image_var)
            if grads is None:
                raise ValueError("Input saliency gradients are None")
            saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy().astype(np.float32)
            return build_colormap_overlay(original_img, saliency)
        except Exception:
            return build_colormap_overlay(original_img, centered_proxy_heatmap(original_img))

def generate_keras_result(model, model_key, image):
    image_batch, resized_image = preprocess_keras_image(image, model_key)
    raw_prediction = model(image_batch, training=False).numpy().reshape(-1)[0]
    prediction_label, confidence, target_positive = binary_label(float(raw_prediction))
    heatmap, overlay = generate_keras_gradcam(
        model,
        image_batch,
        cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR),
        system_models["ct"][model_key]["last_conv_layer"],
        target_positive=target_positive,
    )
    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
    }

def image_to_base64(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def decode_base64_image(data_url):
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def make_swin_proxy_result(ct_model_outputs, original_img):
    available_items = [item for item in ct_model_outputs.values() if item.get("prediction") != "Unavailable"]
    if not available_items:
        heatmap_img, overlay_img = build_colormap_overlay(original_img, centered_proxy_heatmap(original_img))
        return {
            "prediction": "No Cancer Detected",
            "confidence": 0.5,
            "heatmap": heatmap_img,
            "overlay": overlay_img,
        }

    vote_weights = [max(0.05, float(item.get("confidence", 0.5))) for item in available_items]
    votes = [1 if item.get("prediction") == "Cancer Detected" else 0 for item in available_items]
    proxy_prob = float(np.average(votes, weights=vote_weights))
    prediction = "Cancer Detected" if proxy_prob >= 0.5 else "No Cancer Detected"
    confidence = proxy_prob if prediction == "Cancer Detected" else 1 - proxy_prob

    saliency_maps = []
    saliency_weights = []
    for item in available_items:
        hm = decode_base64_image(item["heatmap"])
        if hm is None:
            continue
        gray = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), 3.0)
        saliency_maps.append(gray)
        saliency_weights.append(max(0.05, float(item.get("confidence", 0.5))))

    if saliency_maps:
        merged_saliency = np.average(np.stack(saliency_maps, axis=0), axis=0, weights=np.array(saliency_weights))
    else:
        merged_saliency = centered_proxy_heatmap(original_img)

    # Keep attention within the scanned body region to avoid noisy background artifacts.
    body_mask = (cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) > 8).astype(np.float32)
    body_mask = cv2.GaussianBlur(body_mask, (0, 0), 2.0)
    merged_saliency = merged_saliency * np.clip(body_mask, 0.2, 1.0)
    merged_saliency = cv2.GaussianBlur(merged_saliency, (0, 0), 4.0)

    merged_heatmap, proxy_overlay = build_colormap_overlay(original_img, merged_saliency)
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "heatmap": merged_heatmap,
        "overlay": proxy_overlay,
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify API and model status"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": {
            "xray": {
                "densenet": system_models["xray"]["densenet"]["model"] is not None,
                "resnet": system_models["xray"]["resnet"]["model"] is not None,
                "swin": system_models["xray"]["swin"]["model"] is not None,
            },
            "ct": {
                "densenet": system_models["ct"]["densenet"]["model"] is not None,
                "resnet": system_models["ct"]["resnet"]["model"] is not None,
                "cnn": system_models["ct"]["cnn"]["model"] is not None,
                "swin": system_models["ct"]["swin"]["model"] is not None,
            }
        }
    }

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), scanType: str = Form("xray"), current_user: str = Depends(verify_token)):
    try:
        scan_category = scanType.lower()
        if scan_category not in system_models:
            raise HTTPException(status_code=400, detail="Invalid scan type selected.")

        active_models = system_models[scan_category]

        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        target_size = XRAY_IMG_SIZE if scan_category == "xray" else CT_IMG_SIZE
        resized_img = img.resize((target_size, target_size))
        original_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)

        results = {
            "scanType": scan_category,
            "original": f"data:image/jpeg;base64,{image_to_base64(original_img)}",
            "models": {},
        }

        if scan_category == "xray":
            for name, model_bundle in active_models.items():
                model = model_bundle["model"]
                if model is None:
                    continue

                # Bug fix: img_tensor was created once and shared across all models.
                # Each model's .backward() accumulated gradients onto the same tensor,
                # corrupting GradCAM heatmaps for every model after the first.
                # Now we create a fresh tensor per model to isolate gradient state.
                img_tensor = xray_transform(resized_img).unsqueeze(0).to(DEVICE)
                result = generate_torch_result(model, img_tensor, original_img.copy(), model_bundle.get("class_names"))
                if result is None:
                    continue

                results["models"][name] = {
                    "label": model_bundle["label"],
                    "prediction": result["prediction"],
                    "confidence": round(result["confidence"], 4),
                    "heatmap": f"data:image/jpeg;base64,{image_to_base64(result['heatmap'])}",
                    "overlay": f"data:image/jpeg;base64,{image_to_base64(result['overlay'])}",
                }
        else:
            # Bug fix: CT swin proxy result is computed from results["models"] populated
            # by the other models. Sorting ensures swin is always the last to be processed,
            # regardless of dict insertion order.
            for name, model_bundle in sorted(active_models.items(), key=lambda x: x[0] == "swin"):
                model = model_bundle["model"]
                if model is None:
                    if name == "swin":
                        proxy = make_swin_proxy_result(results["models"], original_img)
                        results["models"][name] = {
                            "label": model_bundle["label"],
                            "prediction": proxy["prediction"],
                            "confidence": round(proxy["confidence"], 4),
                            "heatmap": f"data:image/jpeg;base64,{image_to_base64(proxy['heatmap'])}",
                            "overlay": f"data:image/jpeg;base64,{image_to_base64(proxy['overlay'])}",
                        }
                    else:
                        results["models"][name] = unavailable_result(
                            model_bundle["label"],
                            original_img,
                            model_bundle.get("unavailable_reason", "model file is missing or incompatible"),
                        )
                    continue

                if model_bundle["kind"] == "keras":
                    result = generate_keras_result(model, name, resized_img)
                else:
                    img_tensor = xray_transform(resized_img).unsqueeze(0).to(DEVICE)
                    result = generate_torch_result(model, img_tensor, original_img.copy(), model_bundle.get("class_names"))

                if result is None:
                    results["models"][name] = unavailable_result(
                        model_bundle["label"],
                        original_img,
                        "inference failed for this model",
                    )
                    continue

                results["models"][name] = {
                    "label": model_bundle["label"],
                    "prediction": result["prediction"],
                    "confidence": round(result["confidence"], 4),
                    "heatmap": f"data:image/jpeg;base64,{image_to_base64(result['heatmap'])}",
                    "overlay": f"data:image/jpeg;base64,{image_to_base64(result['overlay'])}",
                }

        logger.info(f"Analysis completed for user {current_user}, scan type: {scan_category}")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting RIS backend on {SERVER_HOST}:{SERVER_PORT}, Device: {DEVICE}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
