import os
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.preprocess import get_data_generators
import matplotlib.pyplot as plt

def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a custom CNN model for Alzheimer's classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a ResNet50-based transfer learning model.
    """
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    DATASET_DIR = "dataset/train/" # Adjusted to point to the subfolder containing class folders
    MODEL_SAVE_PATH = "model/model.h5"
    
    if not os.path.exists("model"):
        os.makedirs("model")

    print("Loading data...")
    train_gen, val_gen = get_data_generators(DATASET_DIR)
    
    # Choose Model Type: Custom CNN or Transfer Learning
    # model = build_cnn_model()
    model = build_transfer_learning_model()
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10 # Use more epochs for real training
    )
    
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Plotting Results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig('model/training_history.png')
    print("Training complete. History plot saved.")
