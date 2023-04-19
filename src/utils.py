import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
def evaluate_models(X_train,X_test,Y_train,Y_test,models,param):
    try:
        # X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)


            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train,Y_train_pred)
            test_model_score = r2_score(Y_test,Y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
        


    except Exception as e :
        raise CustomException(e,sys)