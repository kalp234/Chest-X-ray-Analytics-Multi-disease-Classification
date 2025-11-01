🩺 Chest X-Ray Analytics — Multi-Disease Classification with Explainability

A deep learning–powered web application for multi-label chest disease classification and explainable AI (Grad-CAM visualization) using an ensemble of state-of-the-art CNN and Transformer models.

🚀 Live Overview

Frontend: React.js (interactive upload + explainability UI)

Backend: FastAPI (RESTful inference service)

Models: Ensemble of EfficientNet-B3, EfficientNet-B4, DenseNet-121, and Swin Transformer

Explainability: Grad-CAM heatmaps highlighting disease-relevant lung regions

🧠 Project Highlights

✅ Ensemble Model Integration
Combines multiple architectures (CNN + Transformer) for robust prediction across 14 chest pathologies.

✅ Explainability Framework
Integrated Grad-CAM visualizations to interpret model attention on radiographs.

✅ Optimized Preprocessing Pipeline
Dynamic normalization, resizing, and CLAHE enhancement tailored for radiographic consistency.

✅ Interactive Web Interface
Upload X-rays, view predicted diseases with confidence scores, and visualize pathology-specific heatmaps directly in-browser.

🏗️ System Architecture
React Frontend  →  FastAPI Backend  →  PyTorch Ensemble Models  
      ↑                                   ↓  
   Grad-CAM Heatmap ⟵ Predictions + Probabilities (JSON)
📂 Directory Structure
📦 Predictive-Chest-X-ray-Analytics-Multi-disease-Classification
├── backend/
│   ├── app.py
│   ├── routes/
│   │   ├── inference_routes.py
│   │   └── explain_routes.py
│   ├── utils/
│   │   ├── preprocess.py
│   │   ├── postprocess.py
│   │   └── image_encoder.py
│
├── src/
│   ├── model_loader.py
│   ├── inference_pipeline.py
│   ├── gradcam_utils.py
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
│   │   ├── components/
│   │   │   ├── ImageUploader.js
│   │   │   ├── GradcamOverlay.js
│   │   │   └── PredictionTable.js
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
| Component            | Technology                                         |
| -------------------- | -------------------------------------------------- |
| **Frontend**         | React.js, Axios, CSS3                              |
| **Backend**          | FastAPI, Uvicorn                                   |
| **AI Frameworks**    | PyTorch, Torchvision, TIMM                         |
| **Visualization**    | Grad-CAM, OpenCV, NumPy                            |
| **Model Ensemble**   | EfficientNet-B3/B4, DenseNet-121, Swin Transformer |
| **Explainability**   | Grad-CAM with overlay blending                     |
| **Deployment Ready** | Supports Docker / Local Uvicorn Server             |

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/kalp234/Chest-X-ray-Analytics-Multi-disease-Classification.git
cd Chest-X-ray-Analytics-Multi-disease-Classification
2️⃣ Setup Backend (FastAPI)
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
3️⃣ Setup Frontend (React)
cd frontend
npm install
npm start

🧠 Supported Diseases

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

📊 Model Ensemble Logic
Weighted fusion of predictions:
final_probs = (w1 * eff_b3 + w2 * eff_b4 + w3 * densenet121 + w4 * swin_transformer) / sum(weights)
Grad-CAM overlays are generated dynamically per predicted class.

👨‍💻 Authors
| Name                  | Affiliation             |
| --------------------- | ----------------------- |
| **Kalp Shah**         | BITS Pilani, Goa Campus |
| **Prem Adhiya**       | BITS Pilani, Goa Campus |
| **Ketul Pandya**      | BITS Pilani, Goa Campus |
| **Mohd. Junaid**      | BITS Pilani, Goa Campus |
| **Tanishq Hulyalkar** | BITS Pilani, Goa Campus |
