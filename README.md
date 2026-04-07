
# AI-Enhanced Radiology Information System (RIS)

A comprehensive, full-stack Radiology Information System (RIS) designed for the management, analysis, and diagnostic support of **X-ray** and **CT** imaging. This platform integrates advanced Deep Learning models to provide clinicians with automated diagnostic suggestions and a comparative framework for model performance.

## 📁 Project Structure

```text
├── backend/            # Python API (FastAPI/Flask) & AI Inference
├── frontend/           # React + Vite + Tailwind CSS User Interface
│   ├── src/            # Dashboard, DICOM viewer components, & Analytics
│   ├── index.html
│   └── tailwind.config.js
├── models/             # Model weights (Swin, DenseNet, ResNet)
├── .gitignore          # Excludes node_modules & large weight files
└── README.md
```

## 🚀 Key Features

* **Multi-Modality Support:** Comprehensive workflow management for both **X-ray** and **CT** scans.
* **RIS Workflow Management:** Digitized patient registration, study scheduling, and diagnostic reporting.
* **Comparative AI Benchmarking:** A unique side-by-side evaluation module that compares:
  * **CNN Architectures:** DenseNet and ResNet (optimized for local feature extraction).
  * **Transformer Architectures:** Swin Transformer (optimized for global context).
* **Interactive Analytics:** Visual performance metrics (Accuracy, F1-Score, AUC) for each model across different pathologies.
* **Modern Web Portal:** High-performance interface built with Vite and React for seamless study navigation.

## 🧠 Model Comparison Framework

This system evaluates the strengths of different neural architectures in a clinical setting:

| Architecture | Models Included | Strength in Radiology | 
| ----- | ----- | ----- | 
| **CNN** | DenseNet, ResNet | Excellent at identifying local patterns like small nodules or fine fractures. | 
| **Transformer** | Swin Transformer | Superior at capturing global anatomical relationships in large CT volumes. | 

## 🛠️ Tech Stack

**Frontend:**
* **Framework:** React.js (Vite)
* **Styling:** Tailwind CSS
* **Visualization:** Recharts / Chart.js for model comparison analytics.

**Backend & AI:**
* **Language:** Python 3.9+
* **Deep Learning:** PyTorch / Hugging Face
* **Modalities:** X-ray (2D) and CT (3D/Slice-based) analysis.

## 💻 Getting Started

### Prerequisites

* **Node.js** (v18.0+)
* **Python** (3.9+)
* **GPU Support:** NVIDIA Web Drivers & CUDA (Recommended for CT inference).

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Divyansh010605/your-repo-name.git](https://github.com/Divyansh010605/your-repo-name.git)
   cd your-repo-name
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Backend Setup**
   ```bash
   cd ../backend
   pip install -r requirements.txt
   python main.py
   ```


