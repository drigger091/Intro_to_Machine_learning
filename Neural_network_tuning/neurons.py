step2 = " SELECT THE APPROPIATE NEURONS "
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
from data_prep import create_data

X_train, X_test, Y_train, Y_test = create_data()

def build_model(hp):
    model = Sequential()

    units = hp.Int('units',min_value =8,max_value =128,step =8)
    
    model.add(Dense(units=units,activation='relu',input_dim = 8))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_accuracy', max_trials=5,directory ='mydir',project_name = 'Dipayan')
tuner.search(X_train,Y_train,epochs =5,validation_data =(X_test,Y_test))
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]
#running the model
model.fit(X_train,Y_train,batch_size=32,epochs=100,initial_epoch=6)