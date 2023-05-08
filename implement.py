import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from model import KNN_Classifier

    



def feature_engg(df):
    
    classifier = KNN_Classifier(distance_metric="euclidean") # by default we are using eucildean
    
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    #convert to numpy array
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)
    
    #making sure the X_train also contains the target variable as it is a lazy algorithm no fitting required
    X_train = np.insert(X_train,X_train.shape[1],Y_train,axis=1)
    
    # now using the model to return y_pred for the df
    y_pred = []
    for i in range(X_test.shape[0]): # iterating all the values
        prediction = classifier.predict(X_train,X_test[i],k=5)
        y_pred.append(prediction)
        
    
    return y_pred ,Y_test






