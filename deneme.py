import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from dataModel import Root,Annotation,labels


#%%

# JSON etiketlerini yükle
def load_annotations(json_path):
    try:
        jsonstring=json.load(open(json_path))
        root = Root.from_dict(jsonstring)
        return root

    except:
        print(f"Dosya bulunamadı: {json_path}")
        return None
#%%
def game_time_to_frame(gameTime: str, frame_rate: int = 25) -> int:
    try:
        # Periyot numarasını ve dakika:saniye kısmını ayır
        period, time = gameTime.split(" - ")
        minutes, seconds = map(int, time.split(":"))
        
        # Toplam saniyeyi hesapla
        total_seconds = minutes * 60 + seconds
        
        # Kare numarasını hesapla
        frame_number = total_seconds * frame_rate
        return frame_number
    except Exception as e:
        print(f"Error converting gameTime to frame: {e}")
        return -1  # Hata durumunda -1 döndür

# Örnek kullanım
# game_time = "1 - 97:12"
# current_frame = game_time_to_frame(game_time, frame_rate=25)
# print(f"GameTime '{game_time}' corresponds to frame: {current_frame}")

#%%
def process_video(video_path, annotations, target_labels, total_frames=50000, frame_rate=25):
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    current_frame = 0
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Video uzunluğu (toplam frame sayısı)
    
    # Önemli etiketlerin gameTime bilgilerini topla
    selected_frame_times = []

    # Etiketlerin bulunduğu zaman dilimlerini seçme
    for ann in annotations:
        if ann.label  in target_labels:
            time = game_time_to_frame(ann.gameTime, frame_rate)
            selected_frame_times.append(time)

    # Seçilen frame'lerin sayısını sınırlama
    #selected_frame_times = random.sample(selected_frame_times, min(len(selected_frame_times), total_frames))
    
    # Video karelerini ve etiketlerini işle
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Videoyu baştan başlat
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Eğer bu frame seçilmişse, kareyi ve etiketini kaydet
        if current_frame in selected_frame_times:
            frames.append(cv2.resize(frame, (224, 224)))  # 224x224 boyut
            # Olay var mı kontrol et
            for ann in annotations:
                time = game_time_to_frame(ann.gameTime, frame_rate)
                if current_frame == time:  # Zaman eşleştirme
                    labels.append(ann.label)
                    break
        
        current_frame += 1
        if current_frame >= video_length:
            break

    cap.release()
    
    return np.array(frames), np.array(labels)



#%%
# Veri yükleme ve model eğitimi
Train_video_path = "/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train//england_efl/2019-2020/"
videos=["2019-10-01 - Blackburn Rovers - Nottingham Forest" ,
        "2019-10-01 - Brentford - Bristol City",
        "2019-10-01 - Hull City - Sheffield Wednesday",
        "2019-10-01 - Leeds United - West Bromwich",
        "2019-10-01 - Middlesbrough - Preston North End",
        "2019-10-01 - Reading - Fulham",
        "2019-10-01 - Stoke City - Huddersfield Town"]

#%%
Tlabels = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", 
    "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", 
    "FREE KICK", "GOAL"
]

X= np.empty((0, 224, 224, 3))
y = np.array([], dtype=str)

for name in videos:
    lab=load_annotations(Train_video_path+name+"/Labels-ball.json")
    
    fr,lb=process_video(Train_video_path+name+"/224p.mp4", lab.annotations,Tlabels)
    y = np.concatenate((y, lb), axis=0)
    X = np.concatenate((X, fr), axis=0)
#%%


#%%
# Benzersiz değerler ve frekanslar
unique_values, counts = np.unique(y, return_counts=True)

print("Değerler:", unique_values)
print("Frekanslar:", counts)

#%%
# Çubuk grafiği çizme
plt.figure(figsize=(10, 6))
plt.bar(unique_values, counts, color='skyblue', edgecolor='black')
plt.xlabel('Benzersiz Değerler')
plt.ylabel('Frekans')
plt.title('Benzersiz Değerlerin Frekansı')
plt.xticks(unique_values, rotation=45)  # Değerleri döndürmek için
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Grafik gösterimi
plt.tight_layout()
plt.show()

#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#%%
# Verileri önce eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X / 255, y, test_size=0.2, random_state=42)

# Eğitim setini doğrulama setine de böl
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Şekilleri kontrol et
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape



#%%
print("X_train shape:", X_train.shape)
print("Image ", type(X_train[0]))

#%%
# Rastgele 5 görüntü seç
num_images = 3
random_indices = np.random.choice(X_train.shape[0], num_images, replace=False)  # Rastgele benzersiz indeksler
random_images = X_train[random_indices]
random_lb= y_train[random_indices]
fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
for i, ax in enumerate(axes):
    image = random_images[i]
    
    # Şekil kontrolü: Grayscale mi RGB mi?
    if image.ndim == 2 or image.shape[-1] == 1:
        ax.imshow(image.squeeze(), cmap='gray')  # Grayscale
    else:
        ax.imshow(image)  # RGB
    
    ax.axis('off')
    ax.set_title(f"Image {random_lb[i]}")

plt.tight_layout()
plt.show()


#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

# VGG16 modelini önceden eğitilmiş ağırlıklarla yükleyin
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)

# Katmanları dondurun (transfer learning)
base_model.trainable = False

model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



#%%
# Modelin yapısını dosyaya yazma
# Modeli JSON formatında kaydetme (mimari)
model_json = model.to_json()
with open("model_architecture2.json", "w") as json_file:
    json_file.write(model_json)


#%%
# Modeli eğitme
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler])


#%%
# Dosya yazma işlemi
with open('training_results2.txt', 'w') as file:
    file.write("Epoch\tTraining Loss\tTraining Accuracy\tValidation Loss\tValidation Accuracy\n")
    
    # Eğitim sonuçlarını yazma
    for epoch in range(len(history.history['loss'])):
        file.write(f"{epoch+1}\t{history.history['loss'][epoch]}\t{history.history['accuracy'][epoch]}\t"
                   f"{history.history['val_loss'][epoch]}\t{history.history['val_accuracy'][epoch]}\n")
#%%
# Kayıp grafiği
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss During Training and Validation')
plt.show()
#%%
# Modeli test verisi üzerinde değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
#%%
# Test sonucu değerlendirmesi
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)

# Test sonuçlarını dosyaya yazma
with open('test_results.txt', 'w') as file:
    file.write(f"Test Loss: {test_loss}\n")
    file.write(f"Test Accuracy: {test_accuracy}\n")

#%%
# Doğruluk grafiği
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy During Training and Validation')
plt.show()

#%%
# Test verisi üzerinde tahminler yapmak
y_pred = model.predict(X_test, batch_size=32)

# Tahminlerin ilk 5'ini yazdırma (y_pred'nin her bir elemanı [0, 1, 2, ..., 12] gibi etiketler olabilir)
print("Predictions for the first 5 samples:")
print(np.argmax(y_pred[:5], axis=1))  # One-hot encoded ise argmax ile sınıfı alırız


#%%

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Gerçek etiketler ve tahmin edilen etiketler
y_true = np.argmax(y_test, axis=1)  # One-hot encoded ise argmax ile etiketleri alıyoruz
y_pred_classes = np.argmax(y_pred, axis=1)  # Tahmin edilen sınıflar

# Confusion Matrix hesaplama
cm = confusion_matrix(y_true, y_pred_classes)

# Confusion Matrix görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(12), yticklabels=np.arange(12))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


#%%


# Örnek veri setleri ve etiketler
x_train = np.random.rand(250, 224, 224, 3)  # Eğitim seti
y_train = np.random.randint(0, 12, 250)  # Eğitim etiketleri (10 sınıf)

x_valid = np.random.rand(50, 224, 224, 3)   # Doğrulama seti
y_valid = np.random.randint(0, 12, 50)  # Doğrulama etiketleri (10 sınıf)

x_test = np.random.rand(100, 224, 224, 3)   # Test seti
y_test = np.random.randint(0, 12, 100)  # Test etiketleri (10 sınıf)

# Veri seti boyutları
train_samples = x_train.shape[0]
valid_samples = x_valid.shape[0]
test_samples = x_test.shape[0]
height = x_train.shape[1]  # Tüm setler aynı boyutta olduğu için x_train kullanılabilir
width = x_train.shape[2]
channels = x_train.shape[3]

# Etiket boyutları
train_labels = y_train.shape[0]
valid_labels = y_valid.shape[0]
test_labels = y_test.shape[0]

# Boyutları bir txt dosyasına yazdırma
with open("dataset_dimensions.txt", "w") as f:
    f.write(f"Training set - Number of samples: {train_samples}, Labels: {train_labels}\n")
    f.write(f"Validation set - Number of samples: {valid_samples}, Labels: {valid_labels}\n")
    f.write(f"Test set - Number of samples: {test_samples}, Labels: {test_labels}\n")
    f.write(f"Height: {height}\n")
    f.write(f"Width: {width}\n")
    f.write(f"Channels: {channels}\n")

print("Dataset dimensions have been written to 'dataset_dimensions.txt'")


#%%

#%%

#%%

