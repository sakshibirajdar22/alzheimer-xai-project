import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import load_and_preprocess_image
from xai.saliency import get_saliency_map, overlay_saliency_map

# --- Page Configuration ---
st.set_page_config(
    page_title="Alzheimer's Diagnostic Portal | XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS for Professional Medical UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding-top: 0rem;
    }
    
    /* Custom Header */
    .header-container {
        padding: 2rem;
        background-color: white;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-weight: 700;
        color: #1a3a5f;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #5d7285;
        font-weight: 400;
        font-size: 1.1rem;
    }

    /* Cards */
    .stCard {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #edf2f7;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        padding: 25px;
        border-radius: 20px;
        border-left: 8px solid #007bff;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }
    
    .diag-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    
    .diag-value {
        font-size: 2rem;
        font-weight: 700;
        color: #004085;
    }
    
    .confidence-meter {
        margin-top: 15px;
        font-weight: 600;
        color: #28a745;
    }

    /* Upload Box */
    .upload-container {
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fafc;
        transition: all 0.3s ease;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }

    /* Smooth Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2866/2866321.png", width=80)
    st.title("Admin Console")
    st.markdown("---")
    st.subheader("System Status")
    st.success("🤖 AI Core: Online")
    st.success("🔍 XAI Engine: Active")
    
    st.markdown("---")
    with st.expander("📚 Clinical Definitions"):
        st.write("**Non Demented:** Normal brain function.")
        st.write("**Mild Demented:** Early stage memory loss.")
        st.write("**Moderate Demented:** Significant cognitive decline.")
        st.write("**Very Mild:** Subtle changes, often early detection.")

# --- Header Area ---
st.markdown("""
    <div class="header-container">
        <div class="main-title">🧠 Alzheimer's Diagnostic Portal</div>
        <div class="sub-title">Advanced Explainable AI for Precision Neurology</div>
    </div>
""", unsafe_allow_html=True)

# --- Main Layout ---
MODEL_PATH = "model/model.h5"
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@st.cache_resource
def load_cached_model():
    if os.path.exists(MODEL_PATH):
        # Using compile=False to avoid issues with custom metrics during load
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

def main():
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.subheader("📁 Patient Scan Upload")
        uploaded_file = st.file_uploader("Drop MRI Scan (JPG/PNG)", type=["jpg", "png", "jpeg"], help="Upload a high-resolution T1/T2 weighted MRI slice")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Processed MRI Scan', use_container_width=True)
            
            # Save for XAI
            temp_path = "temp_mri.jpg"
            image.save(temp_path)
        else:
            st.info("👆 Please upload a patient MRI scan to begin clinical analysis.")
            # Placeholder Image Info
            st.image("https://www.researchgate.net/profile/P-Kalaikkumar/publication/343603417/figure/fig1/AS:923594875080704@1597214151703/Sample-MRI-images-from-Kaggle-dataset.png", caption="Example Dataset Scans", use_container_width=True, width=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            with st.status("🩺 Analyzing Neural Patterns...", expanded=True) as status:
                model = load_cached_model()
                time.sleep(1) # Simulate complex analysis
                
                if model:
                    st.write("Extracting feature vectors...")
                    img_array = load_and_preprocess_image(temp_path)
                    
                    st.write("Querying AI Model...")
                    preds = model.predict(img_array, verbose=0)
                    class_idx = np.argmax(preds[0])
                    confidence = preds[0][class_idx] * 100
                    label = CLASS_NAMES[class_idx]
                    status.update(label="Analysis Complete", state="complete")

                    # Display Result Card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="diag-label">Clinical Diagnosis</div>
                        <div class="diag-value">{label}</div>
                        <div class="confidence-meter">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Explainability Section
                    st.divider()
                    st.subheader("💡 Explainability Insights")
                    st.caption("Visualizing specific neural regions that influenced this diagnosis.")
                    
                    with st.spinner("Generating Saliency Maps..."):
                        saliency = get_saliency_map(img_array, model)
                        saliency_img = overlay_saliency_map(temp_path, saliency)
                        st.image(saliency_img, caption='Saliency Highlight (Hotspots of Neural Interest)', use_container_width=True)
                
                else:
                    status.update(label="System Error", state="error")
                    st.error("Model files not found. Ensure `model.h5` is in the `model/` directory.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.subheader("📊 Analysis Engine")
            st.write("The AI analysis engine will provide a diagnosis and visual explanation once a scan is uploaded.")
            st.markdown("""
            ### Steps for Analysis:
            1. Upload a clear MRI scan.
            2. The system pre-processes the image for neural pattern recognition.
            3. Deep Learning determines the Alzheimer's progression stage.
            4. **XAI (Explainable AI)** highlights the biological markers used for detection.
            """)

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey; font-size: 0.8rem;'>© 2026 Alzheimer's XAI Diagnostics | Research Use Only</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
