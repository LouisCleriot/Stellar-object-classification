from src.models.classifier import Classifier
from sklearn.svm import SVC
import scipy 


class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC()
    
    def hyperparameter_tuning(self,X_train,y_train):
        parameters = {
            'C': scipy.stats.expon(scale=100),
            'gamma': scipy.stats.expon(scale=.1),
            'kernel': ['rbf','linear','poly','sigmoid'],
        }
        super().hyperparameter_tuning(X_train,y_train,parameters)
        
