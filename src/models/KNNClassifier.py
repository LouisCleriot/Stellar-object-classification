from src.models.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier(Classifier):

    def __init__(self,name='KNN'):
        super().__init__()
        self.name = name
        self.model = KNeighborsClassifier()