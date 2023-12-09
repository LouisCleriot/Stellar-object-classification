from src.models.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint

class KNNClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'KNN'
        self.model = KNeighborsClassifier()

    def hyperparameter_tuning(self, X_train, y_train):
        parameters = {'n_neighbors': sp_randint(1, 40),
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'weights': ['uniform', 'distance'],
              'p': sp_randint(1, 3),
              'leaf_size': sp_randint(10, 100),
              'n_jobs': [3]
              }
        return super().hyperparameter_tuning(X_train, y_train, parameters, search_type='random', cv=3, scoring='f1_macro', n_iteration=200)