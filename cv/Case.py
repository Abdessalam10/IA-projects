import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#train = pd.read_csv('../../datasets/fashion-mnist_train.csv')
#valid = pd.read_csv('../../datasets/fashion-mnist_test.csv')
#test  = pd.read_csv('../../datasets/fashion-mnist_test.csv')
X_train,y_train = train['features'], train['label']
X_valid,y_valid = valid['features'], valid['label']
X_test,y_test = test['features'], test['label']

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_train_gray = np.sum(X_train/3, axis=3,keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3,keepdims=True)
X_validation_gray = np.sum(X_validation/3, axis=3,keepdims=True)

X_train_gray_norm = (X_train_gray  - 128)/128
X_test_gray_norm = (X_test_gray  - 128)/128
X_validation_gray_norm = (X_validation_gray  - 128)/128

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D,Dropout,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(32,32,1)))
cnn_model.add(AveragePooling2D())
cnn_model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
cnn_model.add(AveragePooling2D())
cnn_model.add(Flatten())
cnn_model.add(Dense(120, activation='relu'))
cnn_model.add(Dense(84, activation='relu'))
cnn_model.add(Dense(43, activation='softmax'))
cnn_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_gray_norm, y_train, epochs=10, validation_data=(X_validation_gray_norm, y_valid))
history= cnn_model.fit(X_train_gray_norm, y_train, batch_size=500, epochs=50, validation_data=(X_validation_gray_norm, y_valid))

scores = cnn_model.evaluate(X_test_gray_norm, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))