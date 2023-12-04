from src.models.Classifier import Classifier
from sklearn.naive_bayes import GaussianNB

class NaiveBayesClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'NaiveBayes'
        self.model = GaussianNB()