from src.models.Classifier import Classifier
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import numpy as np


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "NaiveBayes"
        self.model = GaussianNB()
        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None

    def hyperparameter_tuning(
            self,
            data,
            labels,
            parameters=None,
            search_type="grid",
            cv=5,
            scoring="macro"):
        if parameters is not None:
            print("NaiveBayes does not have hyperparameters to tune \n")
        print(
            "We will test wich bayes classifier is better:"
            "GaussianNB, MultinomialNB, ComplementNB, BernouilliNB and CategoricalNB \n"
        )
        models = [
            GaussianNB(),
            MultinomialNB(),
            ComplementNB(),
            BernoulliNB(),
            CategoricalNB(),
        ]
        rng = np.random.RandomState(0)
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=rng
        )
        for model in models:
            pipeline_steps = [
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "pca_step",
                                PCA(n_components=5),
                                make_column_selector(pattern="u|g|z|r|i"),
                            ),
                            (
                                "redshift_step",
                                FunctionTransformer(),
                                make_column_selector(pattern="redshift"),
                            ),
                        ]
                    ),
                ),
                ("scaler", MinMaxScaler()),
                ("model", model),
            ]
            self.model = Pipeline(pipeline_steps)

            self.model.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            score = f1_score(y_test, y_pred, average=scoring)

            if score > self.best_score:
                self.best_score = score
                self.best_model = self.model
        self.model = self.best_model
        self.model.fit(data, labels)
        print(
            f"The best model is {self.model} with a score of {self.best_score} \n")

    def train(self, data, labels):
        pipeline_steps = [
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "pca_step",
                            PCA(n_components=5),
                            make_column_selector(pattern="u|g|z|r|i"),
                        ),
                        (
                            "redshift_step",
                            FunctionTransformer(),
                            make_column_selector(pattern="redshift"),
                        ),
                    ]
                ),
            ),
            ("scaler", RobustScaler()),
            ("model", self.model),
        ]
        self.model = Pipeline(pipeline_steps)
        self.model.fit(data, labels)
