import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout

from data_prep import create_data


X_train, X_test, Y_train, Y_test = create_data()

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=100,
          validation_data=(X_test, Y_test))

