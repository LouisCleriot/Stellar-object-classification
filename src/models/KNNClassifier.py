from src.models.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint

class KNNClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'KNN'
        self.model = KNeighborsClassifier()

