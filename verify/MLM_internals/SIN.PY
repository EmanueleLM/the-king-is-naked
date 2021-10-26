from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.math import exp
import numpy as np
# load the dataset
dataset = np.random.uniform(0., 3.14*5, 25000)
# split into input (X) and output (y) variables
X = dataset
y = np.sin(dataset) + np.random.rand(len(X))/100
X_valid = np.random.uniform(0., 3.14*5, 1000)
Y_valid = np.sin(X_valid)
# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# fit the keras model on the dataset
model.fit(X, y, epochs=5, batch_size=10, validation_data=(X_valid, Y_valid))

# in-distribution
import matplotlib.pyplot as plt
X_test = np.random.uniform(3.14*0, 3.14*5, 1000)
Y_test = np.sin(X_test)
Y_hat = model.predict(X_test)
plt.scatter(X_test, Y_hat, color='b')
plt.scatter(X_test, Y_test, color='r')
plt.show()

# OOD
import matplotlib.pyplot as plt
X_test = np.random.uniform(3.14*5, 3.14*10, 1000)
Y_test = np.sin(X_test)
Y_hat = model.predict(X_test)
plt.scatter(X_test, Y_hat, color='b')
plt.scatter(X_test, Y_test, color='r')
plt.show()