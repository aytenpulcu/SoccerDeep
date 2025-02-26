#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:13:39 2025

@author: ayten
"""
import json
import cv2
from collections import defaultdict
from dataModel import Root,Train_video_path 

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# JSON etiketlerini yükle
def load_annotations(json_path):
    try:
        jsonstring=json.load(open(json_path))
        root = Root.from_dict(jsonstring)
        return root

    except:
        print(f"Dosya bulunamadı: {json_path}")
        return None


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


def video_sequences( video_name, img_size=(224, 224), num_classes=12):
    """ Belirtilen etiketlere göre CNN-LSTM için uygun X ve y değerlerini döndürür """

    # JSON dosyasını oku
    with open(Train_video_path+video_name+"/Labels-ball.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    video_path = f"{Train_video_path}{video_name}/224p.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  
    print("Bilgi: FrameRate,", frame_rate)

    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı!")
        return None, None

    label_list = defaultdict(list)
    all_labels = []
    X_data, y_data = [], []

    # Etiketleri oku
    for annotation in data["annotations"]:
        label = annotation["label"]

        # Eğer bu etiketten 100'den az varsa listeye ekle
        if len(label_list[label]) < 100:
            current_frame = game_time_to_frame(annotation['gameTime'], frame_rate)
            if current_frame == -1:
                continue  

            # 1 saniye öncesi ve sonrası için frame hesapla
            start_frame = max(0, current_frame - frame_rate)  
            middle_frame = current_frame  
            end_frame = current_frame + frame_rate  
            frames_to_capture = [start_frame, middle_frame, end_frame]

            frames = []
            for frame_no in frames_to_capture:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if ret:
                    # **Resmi yeniden boyutlandır ve normalize et**
                    frame = cv2.resize(frame, img_size)  
                    frame = frame / 255.0  
                    frames.append(frame)

            if len(frames) == 3:  # 3'lü sekans tamamlandıysa ekle
                X_data.append(np.array(frames))  
                y_data.append(label)
                label_list[label].append(frames)  

    cap.release()

    # **Etiketleri Encode et**
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)  
    y_data = to_categorical(y_data, num_classes=num_classes)  

    # **CNN-LSTM için uygun formatta array'e çevir**
    X_data = np.array(X_data)  # (num_samples, timesteps=3, height, width, channels)
    y_data = np.array(y_data)

    print("Hazırlanan veri şekli:", X_data.shape, y_data.shape)
    val_split=0.15
    test_split=0.15
    # Veriyi %70 train, %15 validation, %15 test olarak böl
    X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=(val_split + test_split), random_state=42)
    relative_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_test, random_state=42)

    print(f"Eğitim: {X_train.shape}, Validasyon: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

    
    
    
   
        
            
    