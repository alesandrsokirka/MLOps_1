#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

# Функция для создания данных
def create_data(n_samples, noise=False, anomaly=False):
    time = np.arange(n_samples)
    temperature = 20 + 10 * np.sin(2 * np.pi * time / 365)  # Сезонные колебания
    if noise:
        temperature += np.random.normal(0, 2, n_samples)  # Добавление шума
    if anomaly:
        anomaly_indices = np.random.choice(n_samples, size=5, replace=False)
        temperature[anomaly_indices] += np.random.normal(20, 5, 5)  # Аномалии

    data = pd.DataFrame({'time': time, 'temperature': temperature})
    return data

# Создание директорий для хранения данных
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Создание и сохранение наборов данных
train_data = create_data(1000, noise=True)
test_data = create_data(300, noise=True, anomaly=True)

train_data.to_csv('data/train/train_data.csv', index=False)
test_data.to_csv('data/test/test_data.csv', index=False)


# In[ ]:




