import sklearnex
sklearnex.patch_sklearn()
from src.models.Classifier import Classifier
from sklearn.svm import SVC
import scipy 
import numpy as np


class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC(probability=True)

        
