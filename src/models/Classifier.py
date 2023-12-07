from joblib import dump, load
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from src.helper import plot_roc_curve
import time

class Classifier :
    def __init__(self):
        self.model = None
        self.name = None
        self.best_params = None

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X):
        return self.model.predict(X)

    def hyperparameter_tuning(self,X_train,y_train,parameters,search_type='grid',cv=5,scoring='f1_macro',n_iteration=100):
        
        pipeline_steps = [
            ('preprocess', ColumnTransformer(
                transformers=[
                    ('pca', PCA(n_components=5), make_column_selector(pattern='u|g|z|r|i')),
                    ('scaler', RobustScaler(), make_column_selector(pattern='redshift'))
                ])),
            ('model', self.model)
        ]
        self.model = Pipeline(pipeline_steps)
        parameters = {'model__' + key: value for key, value in parameters.items()}
        
        if pipeline_steps:
            parameters = {'model__' + key: value for key, value in parameters.items()}
            
        if search_type == 'grid':
            self.model = GridSearchCV(self.model,parameters,cv=cv,scoring=scoring)
        elif search_type == 'random':
            self.model = RandomizedSearchCV(self.model,parameters,cv=cv,scoring=scoring)
        elif search_type == 'halving-random':
            self.model = HalvingRandomSearchCV(estimator = self.model, param_distributions=parameters, cv=cv, scoring=scoring)
        
        self.train(X_train,y_train)
        self.best_params = self.model.best_params_
        self.model = self.model.best_estimator_

    def evaluate(self,X,y):
        #classification report
        #calculte inference time
        start_time = time.time()
        y_pred = self.predict(X)
        end_time = time.time()
        inference_time = (end_time - start_time)/len(y_pred)
        print(f'Inference time : {inference_time} seconds')
        print(classification_report(y, y_pred))
        #confusion matrix
        sns.set()
        mat = confusion_matrix(y, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        #roc curve
        plot_roc_curve(self.model, X, y)
        

    def save(self,new_name=None,path='../models/'):
        if new_name == None:
            dump(self.model, f'{path}{self.name}.joblib')  
        else :
            dump(self.model, f'{path}{new_name}.joblib')

    def load(self,new_name=None,path='../models/'):
        if new_name == None:
            self.model = load(f'{path}{self.name}.joblib')
        else :
            self.model = load(f'{path}{new_name}.joblib')