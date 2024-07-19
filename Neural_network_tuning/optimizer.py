"""# steps for tuning the layer
     1) how to select appropiate optimizer
     2) Number of nodes in a layer
     3) how to select the number of hidden layers in a model
     4) all in model( with all the necesary features)"""
     
     
     
step1 = " SELECT THE APPROPIATE OPTIMIZER"
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt

from data_prep import create_data

def build_model(hp):

    model = Sequential()

    model.add(Dense(32,activation='relu',input_dim = 8))
    model.add(Dense(1,activation ='sigmoid'))

    optimizer = hp.Choice('optimizer',values =['adam','sgd','rmsprop','adadelta'])

    model.compile(optimizer=optimizer,loss = 'binary_crossentropy',metrics=['accuracy'])

    return model

tuner = kt.RandomSearch(build_model,objective='val_accuracy',max_trials=5)

X_train, X_test, Y_train, Y_test = create_data()

tuner.search(X_train,Y_train ,epochs =5,validation_data =(X_test,Y_test))


print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]

print(model.summary())
model.fit(X_train,Y_train,batch_size=32,epochs=100,initial_epoch=6,validation_data=(X_test,Y_test))





