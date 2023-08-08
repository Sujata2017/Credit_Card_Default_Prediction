import os
import sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression":LogisticRegression(),
            }

            params = {
                "Random Forest":{
                    "class_weight":["balanced"],
                    "n_estimators":[150,200],
                    'max_depth': [10, 8, 5,20],
                    'min_samples_split': [2, 5, 10],
                },
                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "Logistic Regression":{
                    "class_weight":["balanced"],
                    'penalty': ["l1", "l2", "elasticnet", None],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
                }
            }
           

            #model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)
            
             
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            #logging.info(f"Best found model on both training and testing dataset")
            
            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)

