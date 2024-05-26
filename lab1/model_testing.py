#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка модели и данных
model = joblib.load('model/linear_regression_model.pkl')
test_data = pd.read_csv('data/test/test_data_scaled.csv')

X_test = test_data[['time']]
y_test = test_data['temperature']

# Предсказание и оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error on test data: {mse}')


# In[ ]:




