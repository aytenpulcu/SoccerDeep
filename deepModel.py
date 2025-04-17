#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:19:39 2025

@author: ayten
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def createDeep(frame_height, frame_width, channels, timesteps, num_classes):
# Model Tanımı
    model = Sequential([
    # CNN Katmanları (Özellik Çıkarımı)
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
                    input_shape=(timesteps, frame_height, frame_width, channels)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    
    LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),  
    
    # Tam Bağlantılı Katman (Çıktı)
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),  
    Dense(num_classes, activation='softmax')  # Çok sınıflı tahmin için softmax
])

# Modeli derleme
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
 


