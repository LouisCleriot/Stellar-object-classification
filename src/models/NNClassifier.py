from src.models.Classifier import Classifier
from sklearn.neural_network import MLPClassifier


class NNClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "NeuralNetwork"
        self.model = MLPClassifier()
