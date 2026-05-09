# RIS Setup Guide

## Prerequisites
- Python 3.9+
- Node.js 18+
- pip and npm

## Backend Setup

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the backend directory (copy from `.env.example`):
```bash
cp ../.env.example .env
```

### 4. Initialize Database
The database will be created automatically on first run.

### 5. Run the Server
```bash
python main.py
```
The API will be available at `http://127.0.0.1:8000`

### Health Check
Visit `http://127.0.0.1:8000/api/health` to verify the backend is running.

## Frontend Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run Development Server
```bash
npm run dev
```
The frontend will be available at `http://localhost:5173`

### 3. Build for Production
```bash
npm run build
```

## Model Files

### X-Ray Models
Place model files in `models/XRAY_MODELS/`:
- `densenet_best.pth` - DenseNet169 weights
- `resnet_best.pth` - ResNet50 weights
- `swin_best.pth` - Swin Transformer weights
- `classes.pkl` - Class labels (auto-created)

### CT Scan Models
Place model files in `models/CT_Scan_models/`:
- `densenet121_lung_model.pkl` - DenseNet121 Keras model
- `restnet50.pkl` - ResNet50 Keras model
- `lung_cancer_cnn_model.pkl` - Custom CNN model
- `swin_model.pkl` - Swin Transformer model

## Testing

### Login Credentials
- **Email**: test@ris.local
- **Password**: Test@12345

### Test Image Upload
1. Open `http://localhost:5173` in browser
2. Log in with test credentials
3. Upload an X-ray or CT image
4. Select scan type
5. Click "Run Diagnostics"

## Troubleshooting

### "Models not found" errors
- This is normal if model files aren't downloaded
- The system will gracefully skip unavailable models
- Check `/api/health` endpoint for model status

### CORS errors
- Ensure both frontend and backend are running
- Check API_URL in frontend components

### Database errors
- Delete `users.db` and restart backend to reset
- Or use PostgreSQL by setting `DATABASE_URL`

### Port conflicts
- Change API_URL in frontend components
- Change port in backend/main.py

## Security Notes
- Change `SECRET_KEY` in production
- Use environment variables for sensitive data
- Enable HTTPS in production
- Restrict CORS origins
- Implement rate limiting
