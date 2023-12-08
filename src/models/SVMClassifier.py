from src.models.Classifier import Classifier
from sklearn.svm import SVC
import scipy 
import numpy as np


class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC(probability=True)
    
    def hyperparameter_tuning(self,X_train,y_train):
        parameters = {
            'C': np.logspace(-3, 2, 10),
            'gamma': np.logspace(-3, 2, 10),
            'kernel': ['rbf','linear','poly','sigmoid'],
            'probability': [True],
        }
        super().hyperparameter_tuning(X_train,y_train,parameters,search_type='halving-random')
        
