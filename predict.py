import tensorflow as tf
import numpy as np
from utils.preprocess import load_and_preprocess_image
import os

# Class labels in alphabetical order (as ImageDataGenerator typically loads them)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_alzheimer_stage(image_path, model_path="model/model.h5"):
    if not os.path.exists(model_path):
        return "Model file not found. Please train the model first.", 0
    
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(image_path)
    
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    return CLASS_NAMES[class_idx], confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        label, conf = predict_alzheimer_stage(img_path)
        print(f"Prediction: {label} (Confidence: {conf:.2f}%)")
    else:
        print("Usage: python predict.py <path_to_image>")
