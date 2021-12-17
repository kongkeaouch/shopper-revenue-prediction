import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os
from google.colab import drive

drive.mount('/content/drive')
os.getcwd()
shop = pd.read_csv('drive/kongkea/Dataset/shopper.csv')
shop.head()
shop.describe()
shop.shape
shop.isnull().sum()
plt.figure(figsize=(12, 12))
sns.heatmap(shop.corr(), annot=True, cmap='viridis', linewidths=0.5)

shop.head()
shop['ProductRelated'].value_counts()
shop['Month'].value_counts()
shop['OperatingSystems'].value_counts()
shop['VisitorType'].value_counts()
shop['Browser'].value_counts()
shop['Region'].value_counts()
visitor = pd.get_dummies(shop['VisitorType'])
X = pd.concat([shop, visitor], axis=1)
X.head()
X.columns
y = X['Revenue']
X_new = X.drop(['Revenue', 'Month', 'VisitorType'], axis=1)
X_new['Weekend'] = np.asarray(X_new['Weekend']).astype(np.float32)
y = np.asarray(y).astype(np.float32)
y.shape
X_new.shape
model = Sequential()
model.add(Dense(units=18, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_new, y, epochs=100, validation_split=0.1)
model.save('drive/kongkea/Dataset/Models/shopper_model.h5')
