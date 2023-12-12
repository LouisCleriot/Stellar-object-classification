from src.models.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "KNN"
        self.model = KNeighborsClassifier()
