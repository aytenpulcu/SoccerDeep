from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create3DCNN_BiLSTM(timesteps, height, width, channels, num_classes):
    model = Sequential()

    #  3D CNN (spatio-temporal feature extraction)
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu',
                     padding='same', kernel_regularizer=l2(0.001),
                     input_shape=(timesteps, height, width, channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))  # sadece spatial boyutta küçültme
    model.add(Dropout(0.3))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu',
                     padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu',
                     padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.4))

    #  Zaman boyutunda ayrıştırma (her timestep için özellik vektörü çıkar)
    model.add(TimeDistributed(Flatten()))  # 3D CNN sonrası her timestep için vektör

    #  Bidirectional LSTM
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.3)))

    #  Fully Connected Layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    #  Derleme
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
