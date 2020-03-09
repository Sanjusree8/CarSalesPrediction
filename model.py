import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORT DATASET
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')
 
#TO VISUALIZE THE DATASET 
sns.pairplot(car_df)    

#CREATING TRAINING AND TESTING DATASET/DATA CLEANING 

X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
y = car_df['Car Purchase Amount']
y.shape

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
scaler_x.data_max_
scaler_x.data_min_
print(X_scaled)
y = y.values.reshape(-1,1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)


#TRAINING THE MODEL 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(125, input_dim=5, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs=150, batch_size=25,  verbose=1, validation_split=0.2)



#EVALUATION OF THE MODEL  

print(epochs_hist.history.keys())
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 

# ***(Note that input data must be normalized)***

#X_test_sample = np.array([[0, 50,  500000, 200000, 500000]])
X_test_sample = np.array([[1, 45, 700000, 300000, 600000]])

y_predict_sample = model.predict(X_test_sample)

print('Expected Purchase Amount=', y_predict_sample)
y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)
