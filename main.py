#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:00:04 2025

@author: ayten
"""

from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt


import utils_func
from deepModel import create3DCNN_BiLSTM
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score,recall_score

#%%

# Modeli oluşturulur
# Modeli oluştur

# timesteps=30 
# num_classes=len(Labels)
# frame_height=224 
# frame_width=224
# channels=3
# model = create3DCNN_BiLSTM(timesteps, frame_height, frame_width, channels, num_classes)

# =============================================================================

#kayıtlı modeli tekrar eğit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
model = load_model('cnn_lstm_model2.h5')
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# =============================================================================

#%%


from dataModel import Train_video_path ,videos,Labels
# Global history sözlüğünü başlat
global_history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping

csv_logger = CSVLogger('training_log.csv', append=True)

#  CSV Logger ve EarlyStopping tanımla
csv_logger = CSVLogger('training_log.csv', append=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


#  Her video için artımlı eğitim
for video in videos:
    print(f"\n🎬 {video} için artımlı eğitim başlıyor...")

    # Verileri yükle
    X_train, y_train, X_val, y_val, X_test, y_test = utils_func.video_sequences(video)
    if X_train is None:
        continue
    # Sınıf ağırlıklarını hesaplama
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
    class_weight_dict={i: class_weights[i] for i in range(len(class_weights))}
    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=7,  # her video için epoch
        batch_size=8,
        class_weight=class_weight_dict,
        callbacks=[csv_logger, early_stop],
        verbose=1
    )

    # Global metrik geçmişini kaydet (isteğe bağlı)
    for key in global_history.keys():
        global_history[key].extend(history.history.get(key, []))
    
    # Global değişkenleri kontrol et ve başlat
    if 'X_test_global' not in globals():
        X_test_global, y_test_global = X_test, y_test
    else:
        X_test_global = np.concatenate((X_test_global, X_test), axis=0)
        y_test_global = np.concatenate((y_test_global, y_test), axis=0)

#%%
# Modeli kaydet
model.save("cnn_lstm_model3.h5")

#%%

  # Global test verisiyle nihai değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Global Test Doğruluğu: {test_acc:.2%}")
  

#%%# Model tahmini ve metrik hesaplamaları
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
import matplotlib.pyplot as plt

# Test verisinden tahmin al
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Label adlarını çöz
label_names = utils_func.le.classes_  # LabelEncoder üzerinden sınıf adları

# Confusion Matrix oluştur ve çiz
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=label_names, labels=[0, 1, 2, 3, 4, 5])

print("Classification Report:\n", report)
#%%
# encode değerleri Geri dönüştür 
y_pred_labels = utils_func.le.inverse_transform(y_pred_classes)
y_true_labels = utils_func.le.inverse_transform(y_true_classes)

import os
import datetime

# Kayıt edilecek klasör ve dosya yolu
output_dir = "./sonuclar"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "model_degerlendirme_sonuclari2.txt")

# Zaman damgası (isteğe bağlı)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Dosyaya yaz
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Model Değerlendirme Sonuçları ({timestamp})\n")
    f.write("="*60 + "\n")
    f.write(f"Frame-wise Accuracy: {frame_accuracy:.2f}\n")
    f.write(f"IoU Score: {iou_score:.2f}\n")
    f.write(f"Event-wise Precision: {precision:.2f}, Recall: {recall:.2f}\n")
    f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}\n\n")

    # Classification Report
    f.write("Classification Report:\n")
    f.write(report + "\n")

    # Gerçek ve tahmin edilen label'lar
    f.write("Gerçek Label Değerleri:\n")
    f.write(", ".join(map(str, y_true_labels)) + "\n\n")

    f.write("Tahmin Edilen Label Değerleri:\n")
    f.write(", ".join(map(str, y_pred_labels)) + "\n")