from sklearnex import patch_sklearn
patch_sklearn()
from src.models.Classifier import Classifier
from sklearn.ensemble import RandomForestClassifier

class RFClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'RandomForest'
        self.model = RandomForestClassifier()