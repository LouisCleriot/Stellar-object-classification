from src.models.Classifier import Classifier
from sklearn.linear_model import LogisticRegression

class LogisticRegClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'LogisticRegression'
        self.model = LogisticRegression()