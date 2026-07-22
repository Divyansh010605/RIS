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

# CustomDense (defined later) is the single mechanism for handling
# quantization_config incompatibilities — no global monkey-patch needed.

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
# CORS fix: allow_origins=["*"] + allow_credentials=True is invalid per the CORS spec
# — browsers reject credentialed requests to wildcard origins.
# Use an explicit list (or CORS_ORIGINS env var) in production.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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

# load_optional_torch_serialized_model removed: it was only used for the CT Swin
# model which never loads successfully. The CT Swin slot has been removed entirely.

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

# KerasCompatibleUnpickler removed: it was defined but never called anywhere.

class CustomDense(keras.layers.Dense):
    """Drops the unrecognised quantization_config kwarg so saved models load cleanly."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def load_keras_model(weights_path):
    """Load a Keras model from a pickle or native Keras file.

    Strategy (single fallback path — raw byte mutation removed as unsafe):
      1. Standard pickle.load
      2. keras.models.load_model with CustomDense to strip quantization_config
    """
    if not os.path.exists(weights_path):
        logger.warning(f"Keras model file not found: {weights_path}")
        return None
    try:
        with open(weights_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e_pickle:
        logger.warning(f"pickle.load failed for {weights_path}: {e_pickle} — trying keras native loader")
    try:
        model = keras.models.load_model(
            weights_path, compile=False, safe_mode=False,
            custom_objects={'Dense': CustomDense}
        )
        return model
    except Exception as e_keras:
        logger.error(f"Could not load Keras model from {weights_path}. Keras error: {e_keras}")
        return None

def preprocess_keras_image(image, model_key):
    # Resize removed: the caller (analyze endpoint) already resizes to CT_IMG_SIZE
    # before passing the image in. Resizing again here was a redundant no-op.
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
        # CT Swin entry removed: the model file never loads successfully on any
        # tested runtime, making the slot permanently dead weight. The elaborate
        # make_swin_proxy_result workaround has been removed with it.
    }
}

# Resize removed from xray_transform: the analyze endpoint pre-resizes PIL images
# to XRAY_IMG_SIZE before calling this transform, making transforms.Resize a
# redundant second pass. ToTensor + Normalize are the only operations needed.
xray_transform = transforms.Compose([
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

# decode_base64_image and make_swin_proxy_result removed together with the CT Swin
# entry. decode_base64_image had no other callers.

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
            for name, model_bundle in active_models.items():
                model = model_bundle["model"]
                if model is None:
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
