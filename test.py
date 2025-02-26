#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:35:49 2025

@author: ayten
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)
#%%
import time

# GPU kullanılabiliyor mu?
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Basit bir matris işlemi yapalım
shape = (1000, 1000)
with tf.device(device_name):
    a = tf.random.normal(shape)
    b = tf.random.normal(shape)
    start = time.time()
    c = tf.matmul(a, b)
    end = time.time()

print(f"⏳ İşlem {device_name} üzerinde {end - start:.4f} saniye sürdü")
#%%
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # GPU cihazlarını listele
