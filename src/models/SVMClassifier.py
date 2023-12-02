from src.models.classifier import Classifier
from sklearn.svm import SVC


class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC()
    
    def hyperparameter_tuning(self,X_train,y_train):
        parameters = {
            'kernel':['linear'],
            'C':[0.1,1,10,100],
            'gamma':['scale', 'auto']
        }
        super().hyperparameter_tuning(X_train,y_train,parameters)
        
