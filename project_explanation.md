# 🧠 Alzheimer's XAI Project — Complete Explained Walkthrough

---

## 🌐 Overview: What is this project?

This is a complete, real-world **AI-powered medical system** that:
1. Takes a brain **MRI (Magnetic Resonance Imaging)** scan as input
2. **Classifies** it into one of four Alzheimer's disease stages
3. **Explains** *why* the AI made that decision by highlighting specific brain regions
4. Presents everything in a clean, **doctor-friendly web dashboard**

---

## 🗂️ Project Folder Structure — Explained

```
alzheimer-xai-project/
│
├── dataset/train/         ← Where MRI images live (4 folders)
├── model/                 ← Where the trained model is saved
├── xai/                   ← Explainable AI code
│   ├── gradcam.py         ← (Gradcam attempt — ultimately replaced)
│   ├── saliency.py        ← Actual XAI engine used (Saliency Maps)
│   └── shap_explainer.py  ← SHAP-based explanation (optional/bonus)
├── utils/
│   └── preprocess.py      ← Image loading & dataset loader
├── app/
│   └── app.py             ← Streamlit web application (the UI)
├── train.py               ← Script to train & save the model
├── predict.py             ← Script for CLI-based prediction
└── requirements.txt       ← All Python libraries needed
```

---

## 1. 📁 The Dataset

### What kind of data?
- Brain **MRI images** (greyscale scans of the brain)
- Organized in 4 folders, each folder = 1 class:
  | Folder | Meaning |
  |--------|---------|
  | `NonDemented` | Healthy brain |
  | `VeryMildDemented` | Very subtle memory decline |
  | `MildDemented` | Noticeable brain changes |
  | `ModerateDemented` | Significant Alzheimer's progression |

### Why use MRI Images?
MRI is the **gold standard** for Alzheimer's detection. It shows structural changes in the brain (especially hippocampal shrinkage) long before symptoms appear.

---

## 2. ⚙️ utils/preprocess.py — Data Preparation

### What it does:
This file prepares the raw image data for the AI.

### Key Operations:

**[load_and_preprocess_image()](file:///c:/Users/SAKSHI/Documents/alzheimer-xai-project/utils/preprocess.py#6-16)**
- Opens the image using OpenCV
- Converts colour format (BGR → RGB) because OpenCV loads images as BGR by default but our model expects RGB
- Resizes it to **224×224 pixels** (the standard input size for CNNs)
- Normalizes pixel values from `0-255` → `0.0-1.0` by dividing by 255
- Adds an extra dimension (`expand_dims`) because the model expects a "batch" of images, not just one

**[get_data_generators()](file:///c:/Users/SAKSHI/Documents/alzheimer-xai-project/utils/preprocess.py#17-45)**
- Uses `ImageDataGenerator` from Keras to automatically load thousands of images from folders
- `rescale=1/255` normalizes all images on-the-fly
- `validation_split=0.2` means 20% of images are kept aside for testing (the model never sees these during training)
- Returns two generators: `train_generator` and `val_generator`

### Why normalize to 0–1?
Neural networks learn much better when input values are small. Raw pixel values (0–255) are too large and cause unstable gradients during training.

---

## 3. 🤖 train.py — The Training Engine

This is the most important script. It **teaches** the AI to recognise Alzheimer's.

### Architecture 1: Custom CNN ([build_cnn_model](file:///c:/Users/SAKSHI/Documents/alzheimer-xai-project/train.py#7-31))

```
Input Image (224x224x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU   ← Learns edges & textures
MaxPooling (2x2)                   ← Shrinks image, keeps important features
    ↓
Conv2D (64 filters, 3x3) + ReLU   ← Learns shapes & structures
MaxPooling (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU  ← Learns complex brain patterns
MaxPooling (2x2)
    ↓
Flatten                            ← Converts 3D maps to a 1D list
Dense(128) + ReLU                  ← "Thinking" layer
Dropout(0.5)                       ← Randomly removes neurons (prevents overfitting)
Dense(4, Softmax)                  ← Output: 4 probabilities (one per class)
```

**Why Conv2D?** Convolutional layers apply small filters across the image to detect patterns like edges, curves, and complex shapes. This is how the AI "sees" anatomical structures in the MRI.

**Why MaxPooling?** After detecting features, MaxPooling reduces the size of the feature maps. It keeps only the strongest detected feature in each region, making the model faster and more general.

**Why Dropout?** Dropout randomly switches off 50% of neurons during training. This forces the network to not rely on any single neuron, making it more robust and preventing overfitting (memorizing training data instead of learning).

**Why Softmax?** The output layer has 4 neurons (one per class). Softmax converts the raw outputs into probabilities that all add up to 1.0. The class with the highest probability is the predicted diagnosis.

---

### Architecture 2: ResNet50 (Transfer Learning — What we actually used)

```
Input Image (224x224x3)
    ↓
ResNet50 (Pre-trained on ImageNet) ← Hundreds of learned feature detectors
GlobalAveragePooling2D             ← Condenses features to a single vector
Dense(256) + ReLU
Dropout(0.3)
Dense(4, Softmax)                  ← Output: 4 Alzheimer's stages
```

**Why ResNet50?** ResNet50 is a deep 50-layer neural network that was pre-trained on **ImageNet** (14 million images). It has already learned to detect eyes, edges, textures, shapes — patterns that also appear in MRI scans. We "freeze" the ResNet50 layers (they don't change during training) and only train the final dense layers on our Alzheimer's data. This is called **Transfer Learning**.

**Why Transfer Learning?** Training a 50-layer network from scratch needs millions of MRI images and weeks of compute. Transfer Learning allows us to benefit from those 14 million general images and fine-tune for our specific task with far less data and time.

**Why `include_top=False`?** ResNet50's original top layers were designed to classify 1000 types of objects. We exclude those and replace them with our own 4-class head.

---

### Training Process:

```python
model.fit(train_generator, validation_data=val_generator, epochs=10)
```

- **Epoch**: One full pass through all training images
- **Loss**: How wrong the model is (we use `categorical_crossentropy` for multi-class problems)
- **Accuracy**: Percentage of images classified correctly
- After training, `model.save("model/model.h5")` stores all learned weights to disk

---

## 4. 🔮 predict.py — Standalone Prediction

This is a simple command-line script for quick testing:

```bash
python predict.py path/to/mri_image.jpg
# Output: Mild Demented (Confidence: 87.45%)
```

It loads the saved model, preprocesses the image, and returns the predicted class + confidence score.

---

## 5. 🔍 xai/saliency.py — Explainable AI (The "Why")

This is the **heart of the XAI system**.

### Why do we need XAI?
A regular AI gives you a class label: "Mild Demented." But doctors need to know **which part of the brain** led to that conclusion. Without explanation, the AI is a "black box" — doctors can't trust it clinically.

### How Saliency Maps Work:

1. The image is fed into the model as before
2. But this time, we use `tf.GradientTape()` to track all computations
3. After the prediction, we compute the **gradient of the output class score with respect to the INPUT IMAGE pixels**
4. This gradient tells us: *"If I slightly changed this pixel, how much would the prediction confidence change?"*
5. Pixels with **large gradients = very important** for the diagnosis
6. We color those pixels red/yellow → the resulting visualization is a **Saliency Map**

### Why Saliency Maps instead of Grad-CAM?
Grad-CAM works at the level of a specific internal layer. This caused errors with our ResNet50 model because it's nested inside a Sequential container, making it hard to access internal layers.

Saliency Maps only care about the **input image and the final output**. They work on any model regardless of architecture.

### Clinical Value:
- Red/orange spots = brain regions AI focused on
- If it highlights the **hippocampus** (memory center), that validates the diagnosis
- Doctors can spot whether the AI is looking at the right anatomical features

---

## 6. 🖥️ app/app.py — The Streamlit Web Application

### What is Streamlit?
Streamlit is a Python library that converts Python scripts into interactive web applications without any HTML/CSS/JavaScript knowledge.

### Layout:
- **Left Column**: Patient scan upload area
- **Right Column**: Diagnosis card + XAI visualization
- **Sidebar**: Clinical reference information + system status

### Key Features:

**`@st.cache_resource`** — Loads the model only *once*. Without this, the model would reload on every image upload, which is very slow.

**`st.status()`** — Shows a real-time "Analyzing..." spinner that updates to "Complete" after diagnosis.

**Custom CSS** — We injected CSS to style cards with gradients, shadows, and custom fonts (Inter via Google Fonts):
- `prediction-card`: Diagnosis display with a blue accent border
- `fade-in`: Animation to smoothly reveal results
- `header-container`: Clean white header with drop shadow

### The workflow in the UI:
1. User uploads MRI scan → saved to `temp_mri.jpg`
2. Image is preprocessed (resize, normalize)
3. Model predicts class + confidence
4. Saliency Map is computed and overlaid on the original image
5. Everything is displayed in a clean, animated card

---

## 7. 📦 requirements.txt — The Dependency List

| Library | Purpose |
|---------|---------|
| `tensorflow` | The deep learning framework (runs the neural network) |
| `keras` | High-level API of TensorFlow (makes building models easy) |
| `opencv-python` | Image loading, resizing, and color-map overlays |
| `numpy` | Numerical operations on arrays/tensors |
| `matplotlib` | Plotting training graphs |
| `pandas` | Data handling |
| `scikit-learn` | Confusion matrix and evaluation metrics |
| `streamlit` | Web UI framework |
| [shap](file:///c:/Users/SAKSHI/Documents/alzheimer-xai-project/xai/shap_explainer.py#20-26) | Alternative XAI library (for SHAP explanations) |
| `pillow` | Additional image format support for Streamlit uploader |
| `seaborn` | Statistical visualizations |

---

## 🔄 Full System Flow (End-to-End)

```
[Raw MRI Dataset]
      ↓
  preprocess.py
  (resize → normalize → batch)
      ↓
  train.py
  (CNN or ResNet50)
  (epochs → loss & accuracy)
      ↓
  model/model.h5
  (saved weights)
      ↓  
  app.py (Streamlit)
  ↙              ↘
User Uploads    load_model()
   MRI          (cached)
     ↓
 preprocess
   image
     ↓
  model.predict()
     ↓
 class label +     saliency.py
 confidence     (GradientTape
     ↓           on input)
 Diagnosis Card     ↓
               Saliency Map
                   ↓
           Visual Explanation
```

---

## 💡 Key Concepts Summary

| Concept | Why We Used It |
|---------|---------------|
| ResNet50 | Better accuracy with less training data |
| Transfer Learning | Leverages ImageNet knowledge for MRI domain |
| Dropout | Prevents overfitting |
| Softmax | Converts raw scores to probabilities |
| GradientTape | Tracks computations for Explainable AI |
| Saliency Maps | Explains AI decisions visually without complex layer access |
| Streamlit | Rapid deployment of interactive medical dashboard |
| `@st.cache_resource` | Efficient model loading |

---

## 🏁 Final Take

This project simulates a real-world **clinical AI deployment**:
- It is **accurate** because ResNet50 captures complex brain anatomy patterns
- It is **explainable** because Saliency Maps show exactly what the AI focused on
- It is **usable** because the Streamlit UI is clean, professional, and intuitive for doctors

The model is NOT a replacement for a radiologist — it is a decision support tool that helps doctors detect Alzheimer's **faster and earlier** than conventional methods.
