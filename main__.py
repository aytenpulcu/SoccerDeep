#%%
import subprocess

try:
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
except subprocess.CalledProcessError as e:
    print("Paket yükleme sırasında hata oluştu:", e)
    
#%%
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from dataModel import Root,Annotation,labels,Train_video_path ,videos
from utils import game_time_to_frame,load_annotations,video_sequences
from deepModel import createDeep

#%%


def process_and_save_video(video_name, annotations, Tlabels, frame_rate=25, time_steps=20, save_path="sequences/"):
    video_path = f"{Train_video_path}{video_name}/224p.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Toplam kare sayısı

    # Etiketleri önceden haritala
    label_map = {game_time_to_frame(ann.gameTime, frame_rate): ann.label
                 for ann in annotations if ann.label in Tlabels}

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Kaydedilecek dizin yoksa oluştur

    # Pencereli okuma (Sliding Window)
    for start_frame in range(0, frame_count - time_steps, time_steps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Gereksiz kareleri atla
        
        frames = []
        for _ in range(time_steps):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Boyutu küçülterek bellek yükünü azalt
            frames.append(frame)

        if len(frames) < time_steps:
            break  # Yetersiz frame varsa pencereyi atla

        # Sekansı etiketleme (önemli etiketlere bakarak)
        for idx in range(start_frame, start_frame + time_steps):
            if idx in label_map:
                event_label = label_map[idx]
                break
        else:
            event_label = "NONE"  # Etiket bulunmazsa NONE

        # Sekansı kaydet
        file_name = f"{save_path}{video_name}_sec_{start_frame}_{event_label}.npy"
        np.save(file_name, np.array(frames))  # Sekansı numpy formatında kaydet

    cap.release()


#%%
X = []
y = []

time_steps = 30  # LSTM'nin kullanacağı zaman aralığı
model=createDeep(frame_height, frame_width, channels, timesteps, num_classes)
for name in videos:
    # Modeli Eğit
    X_train, y_train = video_sequences(Train_video_path, name)
    model.fit(X_train, y_train, batch_size=1, epochs=10)


X = np.array(X)  # (num_samples, time_steps, 224, 224, 3)
y = np.array(y)  # (num_samples, )

print("X shape:", X.shape)  # (örneğin) (1000, 30, 224, 224, 3)
print("y shape:", y.shape)  # (örneğin) (1000,)

#%%

def load_sequence(file_path):
    return np.load(file_path)  # NumPy dizisini yükle
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


# #%%
# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X.reshape(X.shape[0], -1), y)

# X = X_resampled.reshape(-1, 224, 224, 3)  # Orijinal şekle geri çevir
# y = y_resampled
#%%
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Etiketleri sayısal değere çevirme
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Kategorik stringleri sayıya çevir
y_one_hot = to_categorical(y_encoded, num_classes=len(labels))  # One-hot formata çevir

# Verileri uygun formata getir
X = X.astype('float32') / 255.0  # Normalizasyon
#%%

#%%


#%%
# Verileri önce eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Eğitim setini doğrulama setine de böl
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

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
# Model için giriş boyutu (frame yüksekliği, genişliği, kanal sayısı)
frame_height = 224
frame_width = 224
channels = 3
timesteps = 30  # LSTM'in işleyeceği zaman adımları
num_classes = 12  # Olay sınıfı sayısı (Örn: Gol, Faul, Korner, Ofsayt, Normal)

model=createDeep(frame_height, frame_width, channels, timesteps, num_classes)
# Model özetini görüntüleme
model.summary()


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
batch = 1  # Her batch 1 maç içerecek
history = model.fit(X_train, y_train, epochs=10, batch_size=batch, validation_data=(X_val, y_val), callbacks=[early_stopping,lr_scheduler])




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

