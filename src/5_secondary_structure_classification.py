import numpy as np
import pandas as pd
import os
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------
# 1. Load labels from 'sample_details.csv'
# --------------------------------------------------------------------
labels_df = pd.read_csv('sample_details_functional.csv')
sample_proteins = labels_df['protein'].values
sample_labels = labels_df['group1'].values
protein_to_label = {protein: label for protein, label in zip(sample_proteins, sample_labels)}

# --------------------------------------------------------------------
# 2. Data generator (for train/validation/test)
# --------------------------------------------------------------------

def data_generator(npz_dir,
                   batch_size,
                   mode='train',
                   shuffle=True):
    """
    Yields (X_batch, y_batch) tuples of shape
      X_batch: (batch_size, 18, 1510354) float32
      y_batch: (batch_size,) int labels

    Args:
      npz_dir: path to 'processed_dataset' parent folder
      batch_size: int
      mode: one of 'train', 'validation', 'test'
      shuffle: whether to shuffle filenames each epoch
    """
    # pick subfolder
    assert mode in ('train','validation','test')
    subdir = os.path.join(npz_dir, mode)
    
    # gather all filenames & labels
    filenames, labels = [], []
    for class_folder in sorted(os.listdir(subdir)):
        if class_folder.startswith('.'): continue
        label = int(class_folder.split('class')[-1]) - 1
        folder_path = os.path.join(subdir, class_folder)
        for fn in os.listdir(folder_path):
            filenames.append(os.path.join(folder_path, fn))
            labels.append(label)
    filenames = np.array(filenames)
    labels    = np.array(labels, dtype=np.int32)
    n_samples = len(filenames)

    # forever loop for Keras
    while True:
        idxs = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(idxs)
        # yield in batches
        for start in range(0, n_samples, batch_size):
            batch_idxs = idxs[start:start+batch_size]
            Xb = []
            for i in batch_idxs:
                sp = load_npz(filenames[i])
                Xb.append(sp.toarray().astype('float32'))
            Xb = np.stack(Xb, axis=0)
            yb = labels[batch_idxs]
            yield Xb, yb

# --------------------------------------------------------------------
# 3. Compute steps per epoch for train/validation/test
# --------------------------------------------------------------------
batch_size = 300
npz_dir = 'processed_dataset/'

def count_samples(split):
    count = 0
    for class_folder in sorted(os.listdir(os.path.join(npz_dir, split))):
        class_path = os.path.join(npz_dir, split, class_folder)
        count += len(os.listdir(class_path))
    return count

train_steps_per_epoch = math.ceil(count_samples('train') / batch_size)
validation_steps_per_epoch = math.ceil(count_samples('validation') / batch_size)
test_steps_per_epoch = math.ceil(count_samples('test') / batch_size)

validation_generator = data_generator(npz_dir, batch_size, mode='validation', shuffle=False)
test_generator = data_generator(npz_dir, batch_size, mode='test', shuffle=False)

# --------------------------------------------------------------------
# 3b. Compute class weights from the train split
# --------------------------------------------------------------------
# Gather all training labels once
train_dir = os.path.join(npz_dir, 'train')
y_train = []
for class_folder in sorted(os.listdir(train_dir)):
    if class_folder.startswith('.'): continue
    label = int(class_folder.split('class')[-1]) - 1
    n = len(os.listdir(os.path.join(train_dir, class_folder)))
    y_train.extend([label] * n)
y_train = np.array(y_train, dtype=np.int32)

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {c: w for c, w in zip(classes, weights)}

print("Computed class_weight:")
for c,w in class_weight.items():
    print(f"  class {c}: {w:.3f}")

# --------------------------------------------------------------------
# 4. Define & compile the model (2-class output)
# --------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(18, 1510354)),
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.1)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(8, activation='softmax', kernel_regularizer=l2(0.1))  # Two classes
])

print(model.summary())

learning_rate = 0.001
optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
rlrop = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=0.0001)
mch = ModelCheckpoint('secondary_structure_batch_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# --------------------------------------------------------------------
# 5. Train the model
# --------------------------------------------------------------------
h = model.fit(
    data_generator(npz_dir, batch_size, mode='train', shuffle=True),
    epochs=50,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps_per_epoch,
    class_weight=class_weight,
    callbacks=[es, rlrop, mch]
)

# --------------------------------------------------------------------
# 6. Evaluate on test set
# --------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps_per_epoch)
print(f'Test accuracy: {test_acc:.4f}')

# --------------------------------------------------------------------
# 7. Generate predictions, print confusion matrix & classification report
# --------------------------------------------------------------------
y_true = []
y_pred = []
y_prob = []

# Step 7a: Accumulate batch-wise results as lists of arrays
for _ in range(test_steps_per_epoch):
    X_batch, y_batch = next(test_generator)
    probs_batch = model.predict(X_batch)         # shape = (batch_size, 2)
    preds_batch = np.argmax(probs_batch, axis=1) # shape = (batch_size,)

    y_true.append(y_batch)       # list of shape-(batch_size,) arrays
    y_pred.append(preds_batch)   # list of shape-(batch_size,) arrays
    y_prob.append(probs_batch)   # list of shape-(batch_size, 2) arrays

# Step 7b: Stack/concatenate into final arrays
y_true = np.concatenate(y_true)     # → shape = (n_samples,)
y_pred = np.concatenate(y_pred)     # → shape = (n_samples,)
y_prob = np.vstack(y_prob)

# Confusion Matrix
conf_raw = confusion_matrix(y_true, y_pred)
print("Raw confusion matrix (counts):")
print(conf_raw)

# 2) Normalize per‐row (so each row sums to 1)
conf_norm = conf_raw.astype(np.float32)
conf_norm = conf_norm / conf_norm.sum(axis=1, keepdims=True)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# --------------------------------------------------------------------
# 8. Plotting: keep first three subplots, replace fourth with ROC curve
# --------------------------------------------------------------------
y_true_bin = label_binarize(y_true, classes=list(range(8)))

fpr = {}
tpr = {}
roc_auc = {}
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))

# Subplot 1: Training & Validation Accuracy
plt.subplot(2, 2, 1)
plt.plot(h.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(h.history['val_accuracy'], color='green', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Subplot 2: Training & Validation Loss
plt.subplot(2, 2, 2)
plt.plot(h.history['loss'], color='red', label='Training Loss')
plt.plot(h.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Subplot 3: Confusion Matrix Heatmap
plt.subplot(2, 2, 3)
sns.heatmap(conf_norm, fmt=".2f", cmap='Blues', 
            xticklabels=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Subplot 4: ROC Curve for Both Classes
plt.subplot(2, 2, 4)
cmap = plt.cm.get_cmap('tab10', 8)

for i in range(8):
    plt.plot(
        fpr[i], tpr[i],
        color=cmap(i),
        lw=2,
        label=f"Class {i} (AUC = {roc_auc[i]:.3f})"
    )

# Random chance diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves (8 Classes)")
plt.legend(loc="lower right", fontsize="small", ncol=2)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results_summary_new_functional.png', dpi=300)  # Save figure with all 4 subplots
plt.show()
