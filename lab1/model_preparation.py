#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка и подготовка данных
train_data = pd.read_csv('data/train/train_data_scaled.csv')

X_train = train_data[['time']]
y_train = train_data['temperature']

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'model/linear_regression_model.pkl')


# In[ ]:




