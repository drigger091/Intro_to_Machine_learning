import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




def create_data():
    
    file_path = 'E:/GITHUBREpo/Intro_to_Machine_learning/Neural_network_tuning/diabetes.csv'

    
    df = pd.read_csv(file_path)
    X = df.drop("Outcome",axis=1)
    Y = df[["Outcome"]]
    
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    
    X_train , X_test ,Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
    
    return X_train ,X_test , Y_train , Y_test


  
    
    