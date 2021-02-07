# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:55:04 2021

@author: User
"""
from flask import Flask, request
import pandas as pd
import numpy as np

app = Flask(__name__)

import pickle
pickle_in = open('new_classifier.pkl','rb')
classifier = pickle.load(pickle_in)




@app.route('/')
def home():
    return "To see the file output go to '/predict_file'"



@app.route('/predict_file', methods = ['GET','POST'])
def predict_file():
    file = request.files.get("file")
    df = pd.read_csv(file)
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df_predict_set = df[['Age','Pclass','Sex']]
    df_predict_set = df.round(2)
    
    #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
    df_predict_set['Age'] = [float(x) for x in list(df_predict_set['Age'])]
    df_predict_set['Pclass'] = [int(x) for x in list(df_predict_set['Pclass'])]
    df_predict_set['Sex'] = [str(x) for x in list(df_predict_set['Sex'])]
    
    
    df_predict_set['sex_female'] = df_predict_set['Sex'].map({'female' : 1, 'male':0})
    df_pclass_d = pd.get_dummies(df_predict_set['Pclass'], prefix='pclass').iloc[:,0:2]
    
    #concatenate the column by axis 1
    df_predict_set = pd.concat([df_predict_set,df_pclass_d],axis=1).reset_index()
    df_predict_set['pclass_1']=df_predict_set['pclass_1'].astype(int)
    df_predict_set['pclass_2'] = df_predict_set['pclass_2'].astype(int)
    df_predict_set['sex_female'] = df_predict_set['sex_female'].astype(int)
    
    
    df_final = df_predict_set[['Age','pclass_1','pclass_2','sex_female']]
    
    #prediction on the file
    try:
        prediction = classifier.predict(df_final)
        return "The predicted value for the csv is: "+ "\n" +"Total number of Outputs "+str(len(prediction)) +"\n" +str(list(prediction))
    except:
        return "Error in prediction file output generation"
    
if __name__=="__main__":
    app.run()
    
    
    
    
    
    
    
    
    