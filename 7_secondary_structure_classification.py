import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import optimizers
from keras.regularizers import l1, l2
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the labels from 'sample_details.csv'
labels_df = pd.read_csv('sample_details_9k.csv')
sample_proteins = labels_df['protein'].values
sample_labels = labels_df['group1'].values

protein_to_label = {protein: label for protein, label in zip(sample_proteins, sample_labels)}

# Function to generate data in chunks
def data_generator(npz_dir, batch_size, mode='train'):
    if mode == 'train':
        npz_dir = os.path.join(npz_dir, 'train')
    elif mode == 'validation':
        npz_dir = os.path.join(npz_dir, 'validation')
    elif mode == 'test':
        npz_dir = os.path.join(npz_dir, 'test')
    
    class_folders = sorted(os.listdir(npz_dir))
    filenames = []
    labels = []
    for class_folder in class_folders:
        if class_folder.startswith('.'):
            continue
        class_label = int(class_folder.split('class')[-1]) - 1
        class_path = os.path.join(npz_dir, class_folder)
        class_files = os.listdir(class_path)
        filenames.extend([os.path.join(class_path, f) for f in class_files])
        labels.extend([class_label] * len(class_files))

    # Convert labels to numpy array
    labels = np.array(labels)

    while True:
        # Shuffle the data for each epoch
        indices = np.arange(len(filenames))
        np.random.shuffle(indices)
        shuffled_filenames = [filenames[i] for i in indices]
        shuffled_labels = labels[indices]

        # Generate data in batches
        for i in range(0, len(filenames), batch_size):
            batch_filenames = shuffled_filenames[i:i+batch_size]
            batch_labels = shuffled_labels[i:i+batch_size]

            data = []
            for filename in batch_filenames:
                sparse_matrix = load_npz(filename)
                dense_matrix = sparse_matrix.toarray().astype(np.float32)
                data.append(dense_matrix)

            yield (np.array(data), batch_labels)

# Define the number of samples and steps per epoch for training
batch_size = 300
npz_dir = 'processed_dataset/'

# Compute the total number of samples and steps per epoch for train set
train_samples_per_class = 0
train_class_folders = sorted(os.listdir(os.path.join(npz_dir, 'train')))
for class_folder in train_class_folders:
    class_path = os.path.join(npz_dir, 'train', class_folder)
    train_samples_per_class += len(os.listdir(class_path))
train_steps_per_epoch = train_samples_per_class // batch_size

# Compute the total number of samples and steps per epoch for validation set
validation_samples_per_class = 0
validation_class_folders = sorted(os.listdir(os.path.join(npz_dir, 'validation')))
for class_folder in validation_class_folders:
    class_path = os.path.join(npz_dir, 'validation', class_folder)
    validation_samples_per_class += len(os.listdir(class_path))
validation_steps_per_epoch = validation_samples_per_class // batch_size

# Compute the total number of samples and steps per epoch for test set
test_samples_per_class = 0
test_class_folders = sorted(os.listdir(os.path.join(npz_dir, 'test')))
for class_folder in test_class_folders:
    class_path = os.path.join(npz_dir, 'test', class_folder)
    test_samples_per_class += len(os.listdir(class_path))
test_steps_per_epoch = test_samples_per_class // batch_size

# Validation data generator
validation_generator = data_generator(npz_dir, batch_size, mode='validation')
test_generator = data_generator(npz_dir, batch_size, mode='test')

# Define the neural network model
model = keras.Sequential([
   keras.layers.Input(shape=(18, 1472442)),
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
   keras.layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.1))
])

# Compile the model
learning_rate = 0.001
optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
rlrop = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=0.0001)
mch = ModelCheckpoint('secondary_structure_batch_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Train the model using the generator
h = model.fit(data_generator(npz_dir, batch_size), epochs=50, steps_per_epoch=train_steps_per_epoch, 
              validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
              callbacks=[es, rlrop, mch])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps_per_epoch)
print(f'Test accuracy: {test_acc}')

# Generate predictions and create confusion matrix
y_true = []
y_pred = []
for _ in range(test_steps_per_epoch):
    X_test_batch, y_test_batch = next(test_generator)
    y_pred_batch = np.argmax(model.predict(X_test_batch), axis=1)
    y_true.extend(y_test_batch)
    y_pred.extend(y_pred_batch)

conf_mat = confusion_matrix(y_true, y_pred, normalize='true')

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(h.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(h.history['val_accuracy'], color='green', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.subplot(2, 2, 2)
plt.plot(h.history['loss'], color='red', label='Training Loss')
plt.plot(h.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(2, 2, 3)
sns.heatmap(conf_mat, annot=False, fmt='d', cmap='Blues', xticklabels=range(3), yticklabels=range(3))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot the classification report
plt.subplot(2, 2, 4)
report = classification_report(y_true, y_pred, output_dict=True)
print("Classification Report:")
print(report)

# Extract unique class labels and convert them to string names
unique_labels = np.unique(y_true)
class_names = [str(label) for label in unique_labels]
print("Class Names:")
print(class_names)

#sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)

metrics = [[report[label]['precision'], report[label]['recall'], report[label]['f1-score']] for label in class_names]
print("Metrics:")
print(metrics)

# Extract accuracy, macro avg, and weighted avg metrics
accuracy = [report['accuracy'], report['accuracy'], report['accuracy']]
macro_avg = [report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']]
weighted_avg = [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']]

# Combine all metrics
all_metrics = metrics + [accuracy, macro_avg, weighted_avg]

# Define row labels
row_labels = class_names + ['accuracy', 'macro avg', 'weighted avg']

# Plot the classification report heatmap
sns.heatmap(pd.DataFrame(all_metrics, columns=['Precision', 'Recall', 'F1-score'], index=row_labels), annot=True, fmt=".3f", cmap='RdBu')

plt.title('Classification Report')
plt.tight_layout()
plt.savefig('results_summary_9k_3rd.png')  # Save the plot to a PNG file
plt.close()
