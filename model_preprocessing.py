#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
train_data = pd.read_csv('data/train/train_data.csv')
test_data = pd.read_csv('data/test/test_data.csv')

# Масштабирование данных
scaler = StandardScaler()

train_data[['temperature']] = scaler.fit_transform(train_data[['temperature']])
test_data[['temperature']] = scaler.transform(test_data[['temperature']])

# Сохранение предобработанных данных
train_data.to_csv('data/train/train_data_scaled.csv', index=False)
test_data.to_csv('data/test/test_data_scaled.csv', index=False)


# In[ ]:




