from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import umap
import pandas as pd
import numpy as np
import click

class DataProcessor :

    def split_data(self,data):
        X = data.drop(['class'], axis=1)
        y = data['class']
        return X,y

    def scale_data(self,X):
        scaler = RobustScaler()
        return scaler.fit_transform(X)
    
    def feature_selection(self,data):
        # Remove features that are not useful
        bad_features = ['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','plate','MJD','fiber_ID']
        return data.drop(bad_features, axis=1)

    def feature_reduction(self,data,reductor='lda'):
        # Transform the features that are highly correlated with dimension reduction
        X,y = self.split_data(data)

        features_to_reduct = ['u','g','r','i','z']
        X_to_reduct = X[features_to_reduct]
        if reductor == 'lda':
            X_reducted = self.lda(X_to_reduct,y)
        elif reductor == 'pca':
            X_reducted = self.pca(X_to_reduct)
        elif reductor == 'umap':
            X_reducted = self.umap(X_to_reduct)
        # Add the reducted features to the dataset
        df_reducted = pd.DataFrame(X_reducted, columns=['reducted_1','reducted_2'])
        # Add original data
        df_reducted['alpha'] = X['alpha']
        df_reducted['delta'] = X['delta']
        df_reducted['class'] = y
        df_reducted.dropna(inplace=True)
        return df_reducted


    def lda(self,X,y):
        lda = LDA(n_components=2)
        lda.fit(X,y)
        return lda.transform(X)
    
    def pca(self,X):
        pca = PCA(n_components=2)
        pca.fit(X)
        return pca.transform(X)
    
    def umap(self,X):
        reducer = umap.UMAP(n_components=2)
        reducer.fit(X)
        return reducer.transform(X)
    
    def remove_outlier(self,data):
        # Remove outliers
        data = data[data['g'] > -2000]
        data = data[data['z'] > -2000]
        data = data[data['u'] > -2000]
        return data
    
    def balance_dataset(self,data):
        X,y = self.split_data(data)

        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X, y)
        df_oversampled = pd.DataFrame(X_sm, columns=data.columns[:-1])
        df_oversampled['class'] = y_sm

        cc = ClusterCentroids(random_state=42)
        X_cc, y_cc = cc.fit_resample(X, y)
        df_undersampled = pd.DataFrame(X_cc, columns=data.columns[:-1])
        df_undersampled['class'] = y_cc

        return df_oversampled, df_undersampled
    
    def process_data(self,data):
        data = self.remove_outlier(data)
        data = self.feature_selection(data)
        X,y = self.split_data(data)
        X = self.scale_data(X)
        df_processed = pd.DataFrame(X, columns=data.columns[:-1])
        df_processed['class'] = y
        df_processed.dropna(inplace=True)
        return df_processed

