"""
train_improved.py  (v2)
-----------------------
Reliable high-accuracy pipeline for Alzheimer's MRI classification.

Key fixes vs previous attempt:
  - Uses MobileNetV2 (lighter, no BN-frozen issue, works great on CPU)
  - Uses tf.keras.applications.mobilenet_v2.preprocess_input (range -1..1)
  - Full model trainable from epoch 1 with a tiny LR (1e-4) so features
    adapt but don't break (no frozen-backbone dead weights)
  - Class weights computed to balance MildDemented/ModerateDemented
  - Label smoothing (0.1) as a regularizer
  - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  - Saves best_model.h5 + overwrites model.h5
  - Saves training plots to model/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR      = "dataset/train/"
MODEL_DIR        = "model"
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 16
NUM_CLASSES      = 4
EPOCHS           = 30        # Increased to 30 for better convergence with regularization

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data generators ───────────────────────────────────────────────────────────
print("Building data generators...")

def mobilenet_preprocess(x):
    return preprocess_input(x)      # scales [0,255] -> [-1,1]

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

class_names = list(train_gen.class_indices.keys())
print(f"Classes     : {class_names}")
print(f"Train images: {train_gen.samples}")
print(f"Val   images: {val_gen.samples}")

# ── Class weights ─────────────────────────────────────────────────────────────
print("\nComputing class weights...")
y_train = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))
print("Weights:", {class_names[k]: round(v, 3) for k, v in class_weight_dict.items()})

# ── Model ─────────────────────────────────────────────────────────────────────
print("\nBuilding MobileNetV2 model (Enhanced Regularization)...")

base = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMG_SIZE, 3),
    alpha=1.0
)
# Unfreeze all layers — we train end-to-end with a very small LR
base.trainable = True

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=True)           # keep BN in training mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5),   # Lowered LR for stability
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
cb = [
    callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training MobileNetV2 with Enhanced Generalization")
print("="*60)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=cb,
    verbose=1
)

# ── Save final ────────────────────────────────────────────────────────────────
# Since restore_best_weights=True, model is now at its best performing state
print(f"\nSaving best performing model -> {FINAL_MODEL_PATH}")
model.save(FINAL_MODEL_PATH)

# ── Plots ─────────────────────────────────────────────────────────────────────
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")

for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')

axes[0].plot(epochs_range, acc,     '#4f9cf9', linewidth=2, label='Train Accuracy')
axes[0].plot(epochs_range, val_acc, '#f97316', linewidth=2, label='Val Accuracy')
axes[0].set_title('Accuracy', color='white', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', color='white')
axes[0].set_ylabel('Accuracy', color='white')
axes[0].legend(facecolor='#1a1d27', labelcolor='white')

axes[1].plot(epochs_range, loss,     '#4f9cf9', linewidth=2, label='Train Loss')
axes[1].plot(epochs_range, val_loss, '#f97316', linewidth=2, label='Val Loss')
axes[1].set_title('Loss', color='white', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', color='white')
axes[1].set_ylabel('Loss', color='white')
axes[1].legend(facecolor='#1a1d27', labelcolor='white')

plt.suptitle('Optimized Training — Enhanced Regularization',
             color='white', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "training_history_improved.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

best_val = max(val_acc)
print(f"\nBest Validation Accuracy : {best_val * 100:.2f}%")
print(f"Training plot saved      -> {plot_path}")
print("\nDone! Run evaluate.py to get the Actual vs Predicted plot.")
