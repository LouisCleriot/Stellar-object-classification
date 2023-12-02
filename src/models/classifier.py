from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Classifier :
    def __init__(self):
        self.model = None
        self.name = None
        self.best_params = None

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X):
        return self.model.predict(X)

    def hyperparameter_tuning(self,X_train,y_train,parameters,search_type='grid',cv=5):
        if search_type == 'grid':
            self.model = GridSearchCV(self.model,parameters,cv=cv)
        elif search_type == 'random':
            self.model = RandomizedSearchCV(self.model,param_distributions=parameters,cv=cv,n_iter=150,random_state=0,n_jobs=-1,verbose=2,scoring='f1_score')
        self.train(X_train,y_train)
        self.best_params = self.model.best_params_
        self.model = self.model.best_estimator_

    def evaluate(self,X,y):
        pred = self.predict(X)
        accuracy = accuracy_score(y,pred)
        precision = precision_score(y,pred,average='weighted')
        recall = recall_score(y,pred,average='weighted')
        f1 = f1_score(y,pred,average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def save(self):
        dump(self.model, f'models/{self.name}.joblib')  

    def load(self):
        self.model = load(f'models/{self.name}.joblib')


    