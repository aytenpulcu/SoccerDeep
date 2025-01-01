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

    # Etiketlerin bulunduğu zaman dilimlerinden rastgele 5000 frame seçme
    for ann in annotations:
        if ann.label  in target_labels:
            time = game_time_to_frame(ann.gameTime, frame_rate)
            selected_frame_times.append(time)

    # Seçilen frame'lerin sayısını sınırlama
    selected_frame_times = random.sample(selected_frame_times, min(len(selected_frame_times), total_frames))
    
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
# Etiketleri one-hot encode yap
def encode_labels(labels, classes):
    label_map = {label: idx for idx, label in enumerate(classes)}
    encoded = np.zeros((len(labels), len(classes)))
    for i, label in enumerate(labels):
        encoded[i, label_map[label]] = 1
    return encoded


#%%

# Model mimarisi
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#%%

# Veri yükleme ve model eğitimi
Train_video_path = "/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train//england_efl/2019-2020/"
videos=["2019-10-01 - Blackburn Rovers - Nottingham Forest" ,"2019-10-01 - Brentford - Bristol City","2019-10-01 - Hull City - Sheffield Wednesday","2019-10-01 - Leeds United - West Bromwich"]
#%%
train =[]
for name in videos:
    ann=load_annotations("/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train//england_efl/2019-2020/"+name+"/Labels-ball.json")
    train+=ann.annotations
#%%
Test_video_path = "/Users/ayten/Documents/SoccerNet/spotting-ball-2024/test/england_efl/2019-2020/2019-10-01 - Reading - Fulham/224p.mp4"  
Valid_video_path = "/Users/ayten/Documents/SoccerNet/spotting-ball-2024/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/224p.mp4"  

test = load_annotations("/Users/ayten/Documents/SoccerNet/spotting-ball-2024/test/england_efl/2019-2020/2019-10-01 - Reading - Fulham/Labels-ball.json")
valid = load_annotations("/Users/ayten/Documents/SoccerNet/spotting-ball-2024/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/Labels-ball.json")
#%%
Tlabels = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", 
    "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", 
    "FREE KICK", "GOAL"
]

train_frames = np.empty((0, 224, 224, 3))
train_labels = np.array([], dtype=str)
for vd in videos:
    fr,lb=process_video(Train_video_path+vd+"/224p.mp4", train,Tlabels)
    train_labels = np.concatenate((train_labels, lb), axis=0)
    train_frames = np.concatenate((train_frames, fr), axis=0)

# Etiketleri encode yap
trainEn_labels = encode_labels(train_labels, labels)
#%%
# Benzersiz değerler ve frekanslar
unique_values, counts = np.unique(train_labels, return_counts=True)

print("Değerler:", unique_values)
print("Frekanslar:", counts)

#%%
test_frames, test_labels = process_video(Test_video_path, test.annotations,labels)
testEn_labels = encode_labels(test_labels, labels)
#%%
valid_frames, valid_labels = process_video(Valid_video_path, valid.annotations,labels)

validEn_labels = encode_labels(valid_labels, labels)

#%%
# Basit veri şekillendirme
X_train = train_frames/255  #
y_train =trainEn_labels

X_test = test_frames/255
y_test= testEn_labels

X_valid = valid_frames/255
y_valid= validEn_labels
#%%
print("X_train shape:", X_train.shape)
print("Image 1 min-max:", X_train[0].min(), X_train[0].max())

#%%
num_images =2
images = X_train[:num_images]

# Görselleri bir döngüde çiz
fig, axes = plt.subplots(1, num_images, figsize=(20,20))
for i, ax in enumerate(axes):
    image = images[i]
    
    # Şekil kontrolü: Grayscale mi RGB mi?
    if image.ndim == 2 or image.shape[-1] == 1:
        ax.imshow(image.squeeze(), cmap='gray')  # Grayscale
    else:
        ax.imshow(image)  # RGB
    
    ax.axis('off')
    ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()

#%%
# Rastgele 5 görüntü seç
num_images = 5
random_indices = np.random.choice(X_train.shape[0], num_images, replace=False)  # Rastgele benzersiz indeksler
random_images = X_train[random_indices]
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
for i, ax in enumerate(axes):
    image = random_images[i]
    
    # Şekil kontrolü: Grayscale mi RGB mi?
    if image.ndim == 2 or image.shape[-1] == 1:
        ax.imshow(image.squeeze(), cmap='gray')  # Grayscale
    else:
        ax.imshow(image)  # RGB
    
    ax.axis('off')
    ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()

#%%


# Modelin tanımlanması
model = models.Sequential([
    # İlk Conv2D katmanı
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    # İkinci Conv2D katmanı
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Üçüncü Conv2D katmanı
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Düzleştirme (Flatten) katmanı
    layers.Flatten(),

    # Tam bağlı (Dense) katman
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Overfitting'i önlemek için dropout katmanı

    # Çıkış katmanı (örneğin, 10 sınıf için softmax)
    layers.Dense(13, activation='softmax')
    # Bu kısmı ihtiyaca göre değiştirebilirsiniz (sınıf sayısı)
])

# Modeli derleme
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Modelin özeti
model.summary()
#%%
# Modelin yapısını dosyaya yazma
# Modeli JSON formatında kaydetme (mimari)
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)


#%%
# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid,y_valid))
#%%
# Dosya yazma işlemi
with open('training_results.txt', 'w') as file:
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(13), yticklabels=np.arange(13))
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

