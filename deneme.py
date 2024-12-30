import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
from typing import List
from typing import Any
from dataclasses import dataclass

@dataclass
class Annotation:
    gameTime: str
    label: str
    position: str
    team: str
    visibility: str

    @staticmethod
    def from_dict(obj: Any) -> 'Annotation':
        _gameTime = str(obj.get("gameTime"))
        _label = str(obj.get("label"))
        _position = str(obj.get("position"))
        _team = str(obj.get("team"))
        _visibility = str(obj.get("visibility"))
        return Annotation(_gameTime, _label, _position, _team, _visibility)

@dataclass
class Root:
    UrlLocal: str
    UrlYoutube: str
    annotations: List[Annotation]

    @staticmethod
    def from_dict(obj: Any) -> 'Root':
        _UrlLocal = str(obj.get("UrlLocal"))
        _UrlYoutube = str(obj.get("UrlYoutube"))
        _annotations = [Annotation.from_dict(y) for y in obj.get("annotations")]
        return Root(_UrlLocal, _UrlYoutube, _annotations)


#%%

# Ana sınıflar
labels = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", 
    "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", 
    "FREE KICK", "GOAL", "NO_ACTION"
]


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
    """
    Converts gameTime string (e.g., "1 - 97:12") to a frame number.
    
    Args:
        gameTime (str): The game time in "1 - MM:SS" format.
        frame_rate (int): Frame rate of the video (default is 25 fps).
    
    Returns:
        int: Corresponding frame number.
    """
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
# Video karelerini çıkar ve etiketlerle eşleştir
def process_video(video_path, annotations, frame_rate=25):
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Kareyi kaydet
        frames.append(cv2.resize(frame, (224, 224)))  # 224x224 boyut
        # Olay var mı kontrol et
        for ann in annotations:
            time=game_time_to_frame(ann.gameTime, frame_rate)
            if current_frame == time:  # Zaman eşleştirme
                labels.append(ann.label)
                break
        else:
            labels.append("NO_ACTION")  # Olay olmayan kareler

        current_frame += 1
        if current_frame== 250:
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

Train_video_path = "C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train/2019-10-01 - Blackburn Rovers - Nottingham Forest/224p.mp4" 
Test_video_path = "C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/test/2019-10-01 - Reading - Fulham/224p.mp4"  
Valid_video_path = "C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/valid/2019-10-01 - Middlesbrough - Preston North End/224p.mp4"  

train = load_annotations("C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/train/2019-10-01 - Blackburn Rovers - Nottingham Forest/Labels-ball.json")
test = load_annotations("C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/test/2019-10-01 - Reading - Fulham/Labels-ball.json")
valid = load_annotations("C:/Users/ayten/Documents/SoccerNet/spotting-ball-2024/valid/2019-10-01 - Middlesbrough - Preston North End/Labels-ball.json")
#%%
train_frames, train_labels = process_video(Train_video_path, train.annotations)

# Etiketleri encode yap
trainEn_labels = encode_labels(train_labels, labels)

test_frames, test_labels = process_video(Test_video_path, test.annotations)
valid_frames, valid_labels = process_video(Valid_video_path, valid.annotations)
testEn_labels = encode_labels(test_labels, labels)
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

num_images =2
images = X_train[:num_images]

# Görselleri bir döngüde çiz
fig, axes = plt.subplots(1, num_images, figsize=(20,20))
#%%
print("X_train shape:", X_train.shape)
print("Image 1 min-max:", X_train[0].min(), X_train[0].max())

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
    layers.Dense(10, activation='softmax')  # Bu kısmı ihtiyaca göre değiştirebilirsiniz (sınıf sayısı)
])

# Modeli derleme
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Modelin özeti
model.summary()

#%%
# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid,y_valid))
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
# Doğruluk grafiği
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy During Training and Validation')
plt.show()

