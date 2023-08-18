# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:18:15 2023

@author: doguk
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
import psutil
try:
    import GPUtil as GPU
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
except:
    gpu = None

#Path
data_dir = '.\HandwrittenNum'

#User-defined parameters
n_epochs = int(input("Enter number of epochs: "))
batch_size = int(input("Enter batch size: "))

#System Monitoring
class SystemMonitor(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Beginning of epoch:", epoch)
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"\nEpoch: {epoch + 1}/{n_epochs}")
        print(f"Time for Epoch: {elapsed_time:.2f} seconds")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram_percent}%")
        if gpu:
            print(f"GPU {gpu.id} {gpu.name}: {gpu.load*100:.1f}%")
            print(f"GPU Memory Used: {gpu.memoryUsed}MB")

#Splitting data
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
#plot a sample image
sample_training_images, _ = next(train_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.imshow(images_arr[0])
    axes.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:1])


#Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
#Additional callback: Reduce learning rate when 'val_loss' has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0004, verbose=1)
#Train the model
model.fit(train_gen, epochs=n_epochs, validation_data=val_gen, callbacks=[SystemMonitor(), model_checkpoint, reduce_lr])

#Predictions
val_gen.reset()
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)

#Performance Metrics
print("\nClassification Report")
print(classification_report(val_gen.classes, y_pred, target_names=val_gen.class_indices.keys()))

print("\nAccuracy Score")
print(accuracy_score(val_gen.classes, y_pred))

print("\nF1 Score")
print(f1_score(val_gen.classes, y_pred, average='weighted'))


