import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

alc = pd.read_csv('../../datasets/Alcohol_Sales.csv')

alc.set_index('DATE', inplace=True)

alc.index = pd.to_datetime(alc.index)
alc.index.freq = 'MS'

alc.rename(columns={'S4248SM144NCEN': 'sales'}, inplace=True)

train = alc.iloc[:len(alc)-12]
test = alc.iloc[len(alc)-12:]

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

gen = TimeseriesGenerator(
    train_scaled,
    train_scaled,
    length=12,
    batch_size=1
)

model = Sequential([
    LSTM(100, activation='relu', input_shape=(12,1)),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    gen,
    epochs=20,
    verbose=1
)

pd.DataFrame(model.history.history).plot()
plt.show()

last_train_batch = train_scaled[-12:]
last_train_batch = last_train_batch.reshape((1, 12, 1))
prediction = model.predict(last_train_batch)
true_prediction = scaler.inverse_transform(prediction)
print(true_prediction)

model.save('TSF_DL_Case.h5')