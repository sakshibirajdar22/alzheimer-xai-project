import os
import numpy as np
import tensorflow as tf
from utils.preprocess import get_data_generators

DATASET_DIR = "dataset/train/"
BEST_PATH = "model/best_model.h5"
FINAL_PATH = "model/model.h5"

def get_acc(path):
    if not os.path.exists(path): return None
    model = tf.keras.models.load_model(path, compile=False)
    _, val_gen = get_data_generators(DATASET_DIR)
    val_gen.reset()
    steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    preds = model.predict(val_gen, steps=steps, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[:len(y_pred)]
    return np.mean(y_pred == y_true)

best_acc = get_acc(BEST_PATH)
final_acc = get_acc(FINAL_PATH)

print(f"BEST_MODEL_ACC: {best_acc*100:.2f}%")
print(f"FINAL_MODEL_ACC: {final_acc*100:.2f}%")

if best_acc and final_acc:
    if best_acc > final_acc:
        print("RESULT: The model peaked and then started declining (overfitting). More epochs won't help without more data or better regularization.")
    else:
        print("RESULT: The model is still improving or stable. More epochs might help slightly.")
else:
    print("RESULT: Could not find model files.")
