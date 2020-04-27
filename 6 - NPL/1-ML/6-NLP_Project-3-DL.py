#!/usr/bin/env python3

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.callbacks import EarlyStopping

from pickle import dump

print('reading input ...')

data_file = pd.read_csv('../../6 - NPL files/yelp_training_set_review(with text_length and transformed)-new.csv')
data_file = data_file[(data_file['stars'] == 1) | (data_file['stars'] == 5)].copy()
data_file = data_file[['stars', 'text_transformed']]
all_text = data_file['text_transformed'].values
all_text = all_text.astype('str')


print('tokenizing ...')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
sequences = tokenizer.texts_to_sequences(all_text)
count_unique = len(tokenizer.word_counts)
pad_encoded = pad_sequences(sequences, maxlen=100, truncating='pre')

X = np.array(pad_encoded)
y = pd.get_dummies(data_file['stars'], drop_first=True).values
seq_len = X.shape[1]

print('construting model ...')

model = Sequential()
model.add(Embedding(count_unique,seq_len)) # rescale the unique number for each token to be small decimal for nomalization; this has to be the first layer
model.add(LSTM(seq_len*2,return_sequences=True))
model.add(LSTM(seq_len*2))
model.add(Dense(50,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


early_stop = EarlyStopping(monitor='val_loss',
                           mode='min',
                           patience=25)

print('model fitting ...')

model.fit(X,y,batch_size=128,epochs=200,verbose=3,
         # callbacks=[early_stop]
         )

print('saving model into h5 ...')

model.save('keras_lstm.h5')


print('dumping tokenizer ...')

dump(tokenizer,open('keras_lstm_tokenizer','wb'))
