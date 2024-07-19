method = " Tune everything at once "
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
from data_prep import create_data

X_train, X_test, Y_train, Y_test = create_data()
method = " Tune everything at once "
def build_model(hp):
    model = Sequential()

    counter = 0

    for i in range (hp.Int("num_layers",min_value =1 ,max_value = 10)):

        if counter == 0:

            model.add(
                Dense(
                hp.Int('units' + str(i),min_value =8,max_value = 128,step =8),
                      activation = hp.Choice('activation'+str(i),values =['relu','tanh','sigmoid']),
                      input_dim = 8
            )
        )
            model.add(Dropout(hp.Choice('Dropout'+ str(i),values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
            
        else:
            model.add(
                Dense(
                hp.Int('units' + str(i),min_value = 8,max_value = 128,step =8),
                activation=hp.Choice('activation_'+str(i) , values=['relu', 'tanh','sigmoid'])
            )
        )
            model.add(Dropout(hp.Choice('Dropout'+ str(i),values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        counter+=1

    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer',values = ['rmsprop','adam','sgd','nadam','adadelta']),
                  loss ='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(build_model,
                        objective='val_accuracy', max_trials=5,
                        directory ='final_all',
                        project_name = 'final')

tuner.search(X_train,Y_train,epochs = 5,validation_data =(X_test,Y_test))

print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]

model.fit(X_train,Y_train,batch_size=32,epochs=200,initial_epoch=6,validation_data=(X_test,Y_test))
