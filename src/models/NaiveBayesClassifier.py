from src.models.Classifier import Classifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

class NaiveBayesClassifier(Classifier):

    def __init__(self,name='NaiveBayes'):
        super().__init__()
        self.name = name
        self.model = GaussianNB()
    
    def hyperparameter_tuning(self, data, labels, parameters=None, search_type='grid', cv=5, scoring='f1_macro'):
        if parameters != None:
            print('NaiveBayes does not have hyperparameters to tune \n')
        print('We will test wich bayes classifier is better: GaussianNB, MultinomialNB, ComplementNB, BernouilliNB and CategoricalNB \n')
        models = [GaussianNB(), MultinomialNB(), ComplementNB(), BernoulliNB(), CategoricalNB()]
        best_score = 0
        best_model = None
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        scaler = MinMaxScaler().fit(X_train)
        X_train_no_negatif = scaler.transform(X_train)
        X_test_no_negatif = scaler.transform(X_test)
        if isinstance(labels[0], str):
            print('Labels are strings, we will encode them \n')
            le = LabelEncoder().fit(y_train)
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
            
        for model in models:
            if isinstance(model, GaussianNB) or isinstance(model, BernoulliNB):
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
            else:
                print(f'Training {model} with no negative values \n')
                model.fit(X_train_no_negatif, y_train)
                score = model.score(X_test_no_negatif, y_test)
            print(f'{model} has score {score} \n')
            if score > best_score:
                best_score = score
                best_model = model
                
        self.model = best_model
        print(f'Best model is {best_model} with score {best_score}, instance has been updated \n')        