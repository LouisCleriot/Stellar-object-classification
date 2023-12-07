from src.models.Classifier import Classifier
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifier(Classifier):

    def __init__(self,name='RandomForest'):
        super().__init__()
        self.name = name
        self.model = RandomForestClassifier()