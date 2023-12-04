from src.models.Classifier import Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

class NaiveBayesClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.name = 'NaiveBayes'
        self.model = GaussianNB()
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def hyperparameter_tuning(self, data, labels, parameters=None, search_type='grid', cv=5, scoring='f1_macro'):
        if parameters != None:
            print('NaiveBayes does not have hyperparameters to tune \n')
        print('We will test wich bayes classifier is better: GaussianNB, MultinomialNB, ComplementNB, BernouilliNB and CategoricalNB \n')
        models = [GaussianNB(), MultinomialNB(), ComplementNB(), BernoulliNB(), CategoricalNB()]
        best_score = 0
        best_model = None
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        if isinstance(labels[0], str):
            print('Labels are strings, we will encode them \n')
            le = LabelEncoder().fit(y_train)
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
            
        for model in models:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_model = model