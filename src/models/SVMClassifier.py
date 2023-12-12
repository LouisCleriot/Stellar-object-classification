from sklearn.svm import SVC
from src.models.Classifier import Classifier
import sklearnex

sklearnex.patch_sklearn()


class SVMClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "SVM"
        self.model = SVC(probability=True)
