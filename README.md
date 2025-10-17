🧠 AI_TraceFinder — Scanner Identification & Tamper Detection
By Nitin Kumar

🚀 Project Overview
AI_TraceFinder is a cutting-edge forensic tool designed to identify the source scanner of a document image and detect tampering or manipulations by analyzing intrinsic device artifacts and texture/frequency signatures.
This project supports digital forensics, document authentication, and legal verification.

🎯 Objectives
Identify the scanner brand/model from intrinsic noise, texture, and compression traces using ML models.
Detect document tampering such as copy-move, retouch, and splicing in scanned images.
Enable authenticity verification for legal and forensic investigations.

🧩 Use Cases
🔍 Digital Forensics: Trace forged or duplicated documents to specific scanners.
🪪 Document Authentication: Differentiate between authorized and unauthorized scanner outputs.
⚖️ Legal Verification: Confirm that scanned documents originate from approved devices.

🏗️ System Overview
Residual Preprocessing:
Haar DWT denoising enhances device/tamper signals for cleaner forensic features.

Scanner Identification:
Hybrid CNN + handcrafted 27-D features
(11 correlation fingerprints + 6 FFT radial energies + 10 LBP-uniform histograms).

Tamper Detection:
Image-level: 18-D patch-averaged features (10 LBP + 6 FFT + 2 contrast stats).
Patch-level fallback: 22-D features (10 LBP + 6 FFT + 3 residual stats + 3 FFT resample stats).
Classification via Calibrated SVM and threshold capping for robustness.

⏱️ Milestones and Timeline
📍 Milestone 1 — Dataset & Preprocessing (Weeks 1–2)
Collect scans from 3–5+ scanner models.
Label dataset → (scanner_model, file_name, page_id).
Preprocess: grayscale → resize 256×256 → Haar DWT denoise → residual generation.

✅ Outputs:
Labeled dataset & manifest CSVs
Verified residual pipeline and sample visuals

📍 Milestone 2 — Feature Engineering & Baselines (Weeks 3–4)
Extract handcrafted features: FFT energies, LBP histograms, PRNU/noise maps.
Train baseline models: Logistic Regression, SVM, Random Forest.
Implement 18-D image-level and 22-D patch-level descriptors.

✅ Outputs:
Baseline metrics & confusion matrices
Visual noise/residual comparisons

📍 Milestone 3 — Deep Model + Explainability (Weeks 5–6)
Train Hybrid CNN (residual image + 27-D vector input).
Evaluate via Accuracy, F1, Confusion Matrix.
Apply Grad-CAM/SHAP for scanner pattern explainability.

✅ Outputs:
scanner_hybrid.keras, scaler.pkl, label_encoder.pkl
Explainability heatmaps & validation reports

📍 Milestone 4 — Deployment & Reporting (Weeks 7–8)
Build Streamlit App:
Upload image → residual + 27-D features → hybrid model predicts scanner & tamper status.
Apply threshold capping for domain consistency.

✅ Outputs:
app.py deployment, documentation & screenshots
Complete system artifacts and comparison reports

⚙️ Methods
🌀 Residual Preprocessing
Grayscale → Resize 256×256 → Haar DWT denoise (zero cH/cV/cD) → Inverse DWT → Residual = Image − Denoised Image

📡 Scanner Identification
27-D Handcrafted Vector:
11× correlations with stored fingerprints
6× FFT radial energies
10× LBP-uniform histograms
Hybrid CNN Input: Residual Image + Standardized 27-D Feature Vector

🔍 Tamper Detection
Image-Level (Preferred):
Per patch 18-D: (10 LBP + 6 FFT + 2 contrast stats [std, mean(|x|−mean)])
SVM classifier with per-domain thresholding
Patch-Level (Fallback):
22-D: (10 LBP + 6 FFT + 3 residual + 3 FFT resample stats)
Aggregation via Top-k Mean + Local-Hit Gating

🧪 Installation
Python 3.10+ required
Install dependencies:
pip install -r requirements.txt

Place Artifacts
Scanner: scanner_hybrid.keras, hybrid_label_encoder.pkl, hybrid_feat_scaler.pkl, scanner_fingerprints.pkl, fp_keys.npy
Tamper Image-Level: image_scaler.pkl, image_svm_sig.pkl, image_thresholds.json
Tamper Patch-Level: patch_scaler.pkl, patch_svm_sig_calibrated.pkl, thresholds_patch.json

▶️ Running the App

Run:
streamlit run app.py

Features:
Upload TIFF/PNG/JPG

Displays:
🧾 Scanner label + confidence
🕵️ Tamper probability + threshold
🧠 Debug Info (domain, scaler dimensions)

📚 Datasets
Scans from multiple scanner models (3–5+), balanced across classes.
Source: In-house scans + public datasets (e.g., Kaggle).

📂 Dataset Drive Link:
Google Drive Folder

📊 Evaluation Metrics
Scanner ID:
Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Robustness across resolution and format
Tamper Detection:
ROC-AUC, Threshold optimization via Youden’s J statistic
Domain thresholds capped for consistent deployment

📈 Results Snapshot:
Patch-Level ROC-AUC ≈ 0.84 (Colab validation)
Tuned image-level thresholds for higher reliability

📁 Repository Layout
app.py                         → Streamlit App  
requirements.txt                → Dependencies  
AI_TraceFinder-FSI.ipynb        → Colab Notebook (Training + Export)  
Ai-Tracefinder/                 → Model & Scaler Artifacts  
Bhagyasri/                      → Complete Project Folder

✅ Milestone Deliverables Checklist
Milestone	Deliverables
M1	Dataset, Preprocessing, Residual Visuals
M2	Feature Extractors, Baselines, Metrics
M3	Hybrid CNN, Explainability, Validation
M4	Streamlit App, Reports, Screenshots

🧭 Troubleshooting
Tamper Output Issues:
Ensure image_thresholds.json is loaded and sidebar shows img_prob, img_thr_raw, img_thr_used.
Feature Dimension Errors:
Scanner scaler: n_features_in_ = 27
Image scaler: n_features_in_ = 18
Patch scaler: n_features_in_ = 22

⚖️ License
Licensed under MIT License.
Respect dataset licenses and cite all upstream resources.

🙌 Acknowledgments
Special thanks to the mentor repository and research community for baseline guidance and forensic insights.
