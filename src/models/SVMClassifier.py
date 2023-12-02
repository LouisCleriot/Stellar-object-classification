from src.models.classifier import Classifier
from sklearn.svm import SVC


class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC(decision_function_shape='ovo')
        
