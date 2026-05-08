"""
evaluate.py
-----------
Evaluates the trained Alzheimer's detection model on the validation set.
Outputs:
  - Classification accuracy (printed to console)
  - model/actual_vs_predicted.png  — bar chart of actual vs predicted counts
  - model/confusion_matrix.png     — heatmap confusion matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.preprocess import get_data_generators

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset/train/"
# Prefer best_model.h5 (saved by improved trainer), fall back to model.h5
MODEL_PATH    = "model/best_model.h5" if os.path.exists("model/best_model.h5") else "model/model.h5"
SAVE_DIR      = "model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# ── Load validation data (shuffle=False to keep label order) ───────────────────
print("Loading validation data...")
_, val_gen = get_data_generators(DATASET_DIR)

class_names = list(val_gen.class_indices.keys())   # e.g. ['MildDemented', ...]
num_classes  = len(class_names)

# ── Predict ────────────────────────────────────────────────────────────────────
print("Running predictions on validation set...")
val_gen.reset()
steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
preds_prob = model.predict(val_gen, steps=steps, verbose=1)

y_pred = np.argmax(preds_prob, axis=1)
y_true = val_gen.classes[: len(y_pred)]            # trim to match prediction count

# ── Accuracy ───────────────────────────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"\n{'='*50}")
print(f"  Validation Accuracy : {acc * 100:.2f}%")
print(f"{'='*50}\n")
print("Per-class report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Plot 1 : Actual vs Predicted (grouped bar chart) ──────────────────────────
actual_counts    = [np.sum(y_true == i) for i in range(num_classes)]
predicted_counts = [np.sum(y_pred == i) for i in range(num_classes)]

x      = np.arange(num_classes)
width  = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars_actual = ax.bar(x - width/2, actual_counts,    width, color="#4f9cf9", label="Actual",    alpha=0.9)
bars_pred   = ax.bar(x + width/2, predicted_counts, width, color="#f97316", label="Predicted", alpha=0.9)

# Value labels on bars
for bar in bars_actual:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', va='bottom', color='white', fontsize=10)
for bar in bars_pred:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', va='bottom', color='white', fontsize=10)

ax.set_xlabel("Alzheimer's Stage", color='white', fontsize=12)
ax.set_ylabel("Number of Samples", color='white', fontsize=12)
ax.set_title(f"Actual vs Predicted — Validation Set\nAccuracy: {acc*100:.2f}%",
             color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(class_names, color='white', fontsize=10)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')

plt.tight_layout()
save_path_bar = os.path.join(SAVE_DIR, "actual_vs_predicted.png")
plt.savefig(save_path_bar, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Actual vs Predicted plot saved -> {save_path_bar}")

# ── Plot 2 : Confusion Matrix heatmap ─────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, linecolor='#333',
            cbar_kws={'shrink': 0.8})

ax.set_xlabel("Predicted Label", color='white', fontsize=12)
ax.set_ylabel("Actual Label",    color='white', fontsize=12)
ax.set_title("Confusion Matrix", color='white', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors='white', labelsize=9)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color='white')
plt.setp(ax.get_yticklabels(), rotation=0,  color='white')

plt.tight_layout()
save_path_cm = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(save_path_cm, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Confusion Matrix saved       -> {save_path_cm}")

print("\nEvaluation complete!")
