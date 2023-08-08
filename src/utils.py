import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report,ConfusionMatrixDisplay


## Save pickel File
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            #pickle.dump(obj, file_obj)
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#def evaluate_models(X_train, y_train,X_test,y_test,models):    
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            #Model Traning
            gs = GridSearchCV(model, para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)


            model.fit(X_train,y_train)


            #make Prediction
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            #train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test,y_test_pred)
            #test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            #report[list(models.values())[i]] = {
            #    "accuracy": test_model_score,
            #    "precision": precision_score(y_test, y_test_pred),
            #    "recall": recall_score(y_test, y_test_pred),
            #    "f1_score": f1_score(y_test, y_test_pred),
            #}

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        
     