# 🩺 Chest X-Ray Analytics — Multi-Disease Classification with Explainability

A deep learning–powered web application for **multi-label chest disease classification** and **explainable AI (Grad-CAM visualization)** using an **ensemble of CNN and Transformer models**.

---

## 🚀 Overview

This project is an **AI-driven diagnostic assistant** designed to analyze chest X-rays and identify multiple thoracic diseases simultaneously.  
It provides predictions along with **visual explanations** using **Grad-CAM heatmaps**, promoting clinical interpretability and model transparency.

---

## 🧠 Project Highlights

✅ **Ensemble Model Integration** — Combines EfficientNet-B3, EfficientNet-B4, DenseNet-121, and Swin Transformer.  
✅ **Explainability Framework** — Grad-CAM overlays highlight the most influential lung regions.  
✅ **Interactive Web Interface** — Built with React.js for smooth image upload and Grad-CAM visualization.  
✅ **Optimized Data Pipeline** — Includes image resizing, normalization, and optional CLAHE enhancement.

---

## 👷 System Architecture

React Frontend → FastAPI Backend → PyTorch Ensemble Models
↑ ↓
Grad-CAM Heatmap ⭟ Predictions + Confidence (JSON)


---

## 📂 Directory Structure

```plaintext
Predictive-Chest-X-ray-Analytics-Multi-disease-Classification
├── backend/
│   ├── app.py
│   ├── routes/
│   │   ├── inference_routes.py
│   │   └── explain_routes.py
│   └── utils/
│       ├── preprocess.py
│       ├── postprocess.py
│       └── image_encoder.py
│
├── src/
│   ├── model_loader.py
│   ├── inference_pipeline.py
│   └── gradcam_utils.py
│
├── models/
│   ├── efficientnet_b3.pth
│   ├── efficientnet_b4.pth
│   ├── densenet121.pth
│   └── swin_transformer.pth
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   └── components/
│   │       ├── ImageUploader.js
│   │       ├── GradcamOverlay.js
│   │       └── PredictionTable.js
│   └── package.json
│
├── data/
│   ├── labels_clean.csv
│   ├── BBox_List_2017.csv
│   └── final_best_probs.npy
│
├── .gitignore
└── README.md

🧩 Technologies Used

| Component         | Technology                    |
| ----------------- | ----------------------------- |
| **Frontend**      | React.js, Axios, TailwindCSS  |
| **Backend**       | FastAPI, Uvicorn              |
| **AI Frameworks** | PyTorch, TorchVision, TIMM    |
| **Visualization** | Grad-CAM, OpenCV, NumPy       |
| **Deployment**    | Docker / Local Uvicorn Server |

⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/kalp234/Chest-X-ray-Analytics-Multi-disease-Classification.git
cd Chest-X-ray-Analytics-Multi-disease-Classification

2️⃣ Backend Setup (FastAPI)
cd backend
python -m venv venv
# Activate virtual environment
venv\Scripts\activate        # for Windows
# source venv/bin/activate   # for macOS / Linux
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000

3️⃣ Frontend Setup (React)
cd frontend
npm install
npm start

🦾 API Endpoints

| Endpoint    | Method | Description                                     |
| ----------- | ------ | ----------------------------------------------- |
| `/predict/` | POST   | Run ensemble inference and return probabilities |
| `/explain/` | POST   | Generate Grad-CAM heatmap for a selected class  |
| `/`         | GET    | Root endpoint / health check                    |

🧠 Supported Diseases (14 Classes)

Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural Thickening
Hernia

📊 Ensemble Strategy
final_probs = (
    w1 * eff_b3 +
    w2 * eff_b4 +
    w3 * densenet121 +
    w4 * swin_transformer
) / sum([w1, w2, w3, w4])

Grad-CAM overlays are generated dynamically for each predicted class.

🧯 Environment Requirements

| Library           | Version |
| ----------------- | ------- |
| Python            | ≥ 3.9   |
| Node.js           | ≥ 18    |
| PyTorch           | ≥ 2.0   |
| FastAPI           | ≥ 0.110 |
| OpenCV/NumPy/TIMM | Latest  |

🧪 Results Summary

| Metric             | Value  |
| ------------------ | ------ |
| **Average AUROC**  | 0.85+  |
| **Macro F1-score** | 0.33   |
| **Inference Time** | ~1.2 s |

🛡️ License
This project is released under the MIT License — free for academic and research use.

👨‍💼 Authors

| Name                  | Affiliation             |
| --------------------- | ----------------------- |
| **Kalp Shah**         | BITS Pilani, Goa Campus |
| **Prem Adhiya**       | BITS Pilani, Goa Campus |
| **Ketul Pandya**      | BITS Pilani, Goa Campus |
| **Mohd. Junaid**      | BITS Pilani, Goa Campus |
| **Tanishq Hulyalkar** | BITS Pilani, Goa Campus |

⭐ Acknowledgements

NIH ChestX-ray14 Dataset
TorchVision & TIMM Libraries
FastAPI Community
Grad-CAM++ & Explainable AI Research
