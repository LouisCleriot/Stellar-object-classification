from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

class Classifier :
    def __init__(self):
        self.model = None
        self.name = None
        self.best_params = None

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X):
        return self.model.predict(X)

    def hyperparameter_tuning(self,X_train,y_train,parameters,search_type='random',cv=5):
        if search_type == 'grid':
            clf = GridSearchCV(self.model,parameters,cv=cv)
        elif search_type == 'random':
            clf = RandomizedSearchCV(self.model,parameters,cv=cv,n_iter=10,random_state=0,n_jobs=-1,verbose=3,scoring='f1')
        clf.fit(X_train,y_train)
        self.best_params = clf.best_params_
        self.model = clf.best_estimator_

    def evaluate(self,X_test,y_test):
        report = classification_report(y_test,self.predict(X_test))
        return report

    def save(self):
        dump(self.model, f'models/{self.name}.joblib')  

    def load(self):
        self.model = load(f'models/{self.name}.joblib')


    