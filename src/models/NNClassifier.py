from src.models.Classifier import Classifier
from sklearn.neural_network import MLPClassifier

class NNClassifier(Classifier):

    def __init__(self,name='NeuralNetwork'):
        super().__init__()
        self.name = name
        self.model = MLPClassifier()