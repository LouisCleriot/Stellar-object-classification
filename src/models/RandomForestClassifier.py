from sklearn.ensemble import RandomForestClassifier
from src.models.Classifier import Classifier
from sklearnex import patch_sklearn

patch_sklearn()


class RFClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "RandomForest"
        self.model = RandomForestClassifier()
