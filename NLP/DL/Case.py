import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sms=pd.read_csv('../../datasets/spam.csv', encoding='ISO-8859-1')

sms.info()

#preprocessing

sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
sms.rename(columns={'v1':'label', 'v2':'message'}, inplace=True)
y=sms['label'].map({'ham':1, 'spam':0})
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sms['message'], y, test_size=0.2, random_state=0)
print(x_train.shape)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train_pad = pad_sequences(x_train_seq)
x_test_pad = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=x_train_pad.shape[1])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPool1D

input_layer = Input(shape=(x_train_pad.shape[1],))
x = Embedding(len(tokenizer.word_index) + 1, 20)(input_layer)
x=LSTM(15, return_sequences=True)(x) #“Understand context through time.”
x=GlobalMaxPool1D()(x) #“Extract the most important features from the sequence.”
x=Dense(1, activation='sigmoid')(x) #  “Use these strongest signals to classify.”                                                          
model = Model(input_layer, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_pad, y_train, epochs=20, validation_data=(x_test_pad, y_test))

pd.DataFrame(model.history.history).plot()
plt.show()