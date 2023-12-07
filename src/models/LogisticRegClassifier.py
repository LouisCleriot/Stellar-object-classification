from src.models.Classifier import Classifier
from sklearn.linear_model import LogisticRegression

class LogisticRegClassifier(Classifier):

    def __init__(self,name='LogisticRegression'):
        super().__init__()
        self.name = name
        self.model = LogisticRegression()