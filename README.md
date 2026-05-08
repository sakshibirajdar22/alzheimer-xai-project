# Alzheimer’s Disease Detection using Explainable AI (XAI)

This project provides a complete end-to-end healthcare system for detecting Alzheimer's Disease stages from MRI brain images using Deep Learning and Explainable AI.

## 🚀 Features
- **Classification**: Detects 4 stages: NonDemented, VeryMildDemented, MildDemented, ModerateDemented.
- **Explainable AI (XAI)**: Includes **Grad-CAM** heatmaps to visualize brain regions contributing to the diagnosis.
- **Modern UI**: Professional medical-style dashboard built with **Streamlit**.
- **Transfer Learning**: Supports ResNet50 for high-accuracy performance.

## 🛠️ Tech Stack
- **Languages**: Python
- **Frameworks**: TensorFlow / Keras, Streamlit
- **Computer Vision**: OpenCV, Matplotlib
- **XAI Libraries**: SHAP, Grad-CAM (Custom implementation)

## 📂 Project Structure
```
project/
│
├── dataset/                # Place MRI images here (4 subfolders)
├── model/                  # Saved models and training plots
├── xai/                    # XAI Modules (Grad-CAM, SHAP)
├── utils/                  # Preprocessing utilities
├── app/                    # Streamlit application
├── train.py                # Model training script
├── predict.py              # CLI prediction script
├── requirements.txt        # Dependencies
└── README.md
```

## ⚙️ Installation



2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**:
   Organize your dataset in the `dataset/` folder:
   - `dataset/NonDemented/`
   - `dataset/VeryMildDemented/`
   - `dataset/MildDemented/`
   - `dataset/ModerateDemented/`

## 📈 Training
Run the training script to build and save the model:
```bash
python train.py
```
This will save the model to `model/model.h5` and training graphs to `model/training_history.png`.

## 🖥️ Running the App
Launch the Streamlit dashboard:
```bash
streamlit run app/app.py
```

## 🔍 Explainability
The application uses **Grad-CAM** to overlay a heatmap on the MRI image. Red/Warm regions indicate high importance areas that the model used to determine the Alzheimer's stage.

---
*Disclaimer: This is for educational and research purposes only. Always consult medical professionals for actual diagnosis.*
