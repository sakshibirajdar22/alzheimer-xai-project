import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image and preprocesses it for model prediction (MobileNetV2 style).
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32')
    img = preprocess_input(img)      # scales to [-1, 1]
    img = np.expand_dims(img, axis=0)
    return img

def get_data_generators(dataset_dir, target_size=(224, 224), batch_size=32):
    """
    Creates train and validation data generators with MobileNetV2 preprocessing.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator
