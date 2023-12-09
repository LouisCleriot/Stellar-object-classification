from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import umap
import pandas as pd
import numpy as np
import click
#import logging
from halo import Halo



class DataProcessor :

    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=5)
        self.encoder = LabelEncoder()
        
    def split_data(self,data):
        X = data.drop(['class'], axis=1)
        y = data['class']
        return X,y

    def scale_data(self,data,fit=False):
        X,y = self.split_data(data)
        if fit:
            self.scaler.fit(X)
        new_X = self.scaler.transform(X)
        df = pd.DataFrame(new_X, columns=X.columns)
        df['class'] = y
        return df
    
    def feature_selection(self,X,features):
        # Keep only the features that are in the list
        X = X[features]
        return X

    def remove_correlation(self,data,fit=False):
        """
        Remove the correlation between features using PCA.
        If fit is True, the pca is fitted with the data. In this case,the data should be used for training only and not for testing.
        """
        X,y = self.split_data(data)
       
        if fit :
            new_X = self.pca.fit(X[['u','g','r','i','z']])

        # Transform data
        new_X = self.pca.transform(X[['u','g','r','i','z']])

        # Add the reducted features to the new dataset
        df = pd.DataFrame(new_X, columns=['u','g','r','i','z'])
        df.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        # Add original data 
        df['redshift'] = data['redshift']
        df['class'] = data['class']
       
        return df
    
    def oversample_data(self,data):
        X,y = self.split_data(data)
        spinner = Halo(text='oversampling the data', spinner='dots')
        spinner.start()
        sm = SMOTE(random_state=0)
        X_sm, y_sm = sm.fit_resample(X, y)
        df_oversampled = pd.DataFrame(X_sm, columns=X.columns)
        df_oversampled['class'] = y_sm
        spinner.succeed('data oversampled')
        return df_oversampled
    
    def undersample_data(self,data):
        X,y = self.split_data(data)
        #spinner = Halo(text='undersampling the data', spinner='dots')
        #spinner.start()
        rus = RandomUnderSampler(random_state=0)
        X_rus, y_rus = rus.fit_resample(X, y)
        df_undersampled = pd.DataFrame(X_rus, columns=X.columns)
        df_undersampled['class'] = y_rus
        #spinner.succeed('data undersampled')
        return df_undersampled
    
    def encode_labels(self,data,fit=False):
        X,y = self.split_data(data)
        if fit:
            self.encoder.fit(y)
        new_y = self.encoder.transform(y)
        df = pd.DataFrame(X, columns=X.columns)
        df['class'] = new_y
        return df
    
    

    
