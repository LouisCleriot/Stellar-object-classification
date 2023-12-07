from src.models.Classifier import Classifier
from sklearn.svm import SVC


class SVMClassifier(Classifier):

    def __init__(self,name='SVM'):
        super().__init__()
        self.name = name
        self.model = SVC()
     
