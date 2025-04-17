#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:00:04 2025

@author: ayten
"""

from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt

from dataModel import Train_video_path ,videos
import utils_func
from deepModel import createDeep

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score,recall_score
#%%
# class DataGenerator(Sequence):
#     def __init__(self, video_list, batch_size, timesteps, frame_height, frame_width, channels, num_classes):
#         self.video_list = video_list
#         self.batch_size = batch_size
#         self.timesteps = timesteps
#         self.frame_height = frame_height
#         self.frame_width = frame_width
#         self.channels = channels
#         self.num_classes = num_classes

#     def __len__(self):
#         return int(np.floor(len(self.video_list) / self.batch_size))

#     def __getitem__(self, index):
#         batch_videos = self.video_list[index * self.batch_size:(index + 1) * self.batch_size]
#         X, y = self.__data_generation(batch_videos)
#         return X, y

#     def __data_generation(self, batch_videos):
#         X_batch, y_batch = [], []
#         for video in batch_videos:
#             X, y = utils_func.video_sequences( video)  # Video'yu işle
#             X_batch.append(X)
#             y_batch.append(y)

#         X_batch = np.array(X_batch)
#         y_batch = np.array(y_batch)

#         return X_batch, y_batch
#%%
# Modeli oluşturulur

model=createDeep(frame_height=224, frame_width=224, channels=3, timesteps=30, num_classes=12)
#%%


# Global test verilerini saklamak için listeler
X_test_global = []
y_test_global = []
# Global history sözlüğünü başlat
global_history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger('training_log.csv', append=True)

# Her video için artımlı eğitim
for video in videos:
    print(f"\n📌 {video} üzerinde artımlı eğitim başlıyor...")
    X_train, y_train, X_val, y_val, X_test, y_test = utils_func.video_sequences( video)
    
    if X_train is None:
        continue
    # Sınıf ağırlıklarını hesaplama
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
    class_weight_dict={i: class_weights[i] for i in range(len(class_weights))}
     # Artımlı eğitim: Her video için belirli sayıda epoch
    history=model.fit(X_train, y_train, epochs=4, batch_size=8, validation_data=(X_val, y_val), class_weight=class_weight_dict,callbacks=[csv_logger])
    
    # Global history sözlüğünü güncelle 
    for key in global_history.keys():
        global_history[key].extend(history.history[key])
    # Global test verisini birleştir
    if X_test_global and y_test_global:
        X_test_global = np.concatenate(X_test, axis=0)
        y_test_global = np.concatenate(y_test, axis=0)

        

# Modeli kaydet
model.save("cnn_lstm_model.h5")

#%%

  # Global test verisiyle nihai değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Global Test Doğruluğu: {test_acc:.2%}")
  
# %%

# Model tahmini ve metrik hesaplamaları
y_pred = model.predict(X_train)
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_train, axis=-1)

# Frame-wise Accuracy Hesaplama
frame_accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Frame-wise Accuracy: {frame_accuracy:.2f}")

# IoU (Intersection over Union) Hesaplama
def calculate_iou(true_labels, pred_labels):
    intersection = np.logical_and(true_labels, pred_labels).sum()
    union = np.logical_or(true_labels, pred_labels).sum()
    return intersection / union if union > 0 else 0

iou_score = calculate_iou(y_true_classes.flatten(), y_pred_classes.flatten())
print(f"IoU Score: {iou_score:.2f}")

# Event-wise Precision ve Recall Hesaplama
precision = precision_score(y_true_classes.flatten(), y_pred_classes.flatten(), average='macro')
recall = recall_score(y_true_classes.flatten(), y_pred_classes.flatten(), average='macro')
print(f"Event-wise Precision: {precision:.2f}, Recall: {recall:.2f}")
#%%
# Grafik çizme fonksiyonu
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)

    # Kayıp (Loss) Grafiği
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Eğitim Kaybı')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()

    # Doğruluk (Accuracy) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Eğitim Doğruluğu')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Doğrulama Doğruluğu')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.legend()

    plt.show()

# Eğitim sürecini görselleştir
plot_training_history(history)

#%%

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Global test verisi (X_test_global, y_test_global) modelin daha önce toplandığını varsayıyoruz.
# Modelin global test verisi üzerindeki performansını ölçelim:
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

# Tahminleri al
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix oluştur
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred)
print("Classification Report:\n", report)
