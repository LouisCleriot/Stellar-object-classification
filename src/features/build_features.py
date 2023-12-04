from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import umap
import pandas as pd
import numpy as np
import click
import logging

class Reductor :
    def __init__(self,method='umap',n=2):
        self.name = method
        if method == 'lda':
            self.method = LDA(n_components=2)
        elif method == 'pca':
            self.method = PCA(n_components=n)
        else :
            self.method = umap.UMAP(n_components=n)

    def fit(self,X,y):
        if self.name == 'lda':
            self.method.fit(X,y)
        else:
            self.method.fit(X)

    def transform(self,X):
        return self.method.transform(X)


class DataProcessor :

    def __init__(self,reductor='lda'):
        self.scaler = RobustScaler()
        self.reductor1 = Reductor(reductor,n=3)
        self.reductor2 = Reductor(reductor,n=2)
        

    def split_data(self,data):
        X = data.drop(['class'], axis=1)
        y = data['class']
        return X,y

    def scale_data(self,X,fit=False):
        if fit:
            self.scaler.fit(X)
        return self.scaler.transform(X)
    
    def feature_selection(self,X,features):
        # Keep only the features that are in the list
        X = X[features]
        return X

    def feature_reduction(self,data,fit=False):
        """
        Reduce the dimension of the dataset using the reductor(PCA, LDA or UMAP).
        If fit is True, the reductor is fitted with the data. In this case,the data should be used for training only and not for testing.
        """
        X,y = self.split_data(data)

        # Transform the features that are highly correlated with dimension reduction
        features_to_reduct_1 = ['u','g','z']
        features_to_reduct_2 = ['r','i']
        X1_to_reduct = X[features_to_reduct_1]
        X2_to_reduct = X[features_to_reduct_2]
        if fit:
            self.reductor1.fit(X1_to_reduct,y)
            self.reductor2.fit(X2_to_reduct,y)
        X1_reducted = self.reductor1.transform(X1_to_reduct)
        X2_reducted = self.reductor2.transform(X2_to_reduct)

        # Reconstruction of the dataframe
        X_reducted = np.concatenate((X1_reducted,X2_reducted),axis=1)

        # Add the reducted features to the new dataset
        df_reducted = pd.DataFrame(X_reducted, columns=['ugz_1','ugz_2','ugz_3','ri_1','ri_2'])

        # Add original data 
        df_reducted['redshift'] = X['redshift']
        df_reducted['class'] = y
        df_reducted.dropna(inplace=True)
        return df_reducted
    
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
    
    def process_data(self,data,train=False):
        X,y = self.split_data(data) 
        # if data is for training, fit the scaler
        X = self.scale_data(X,fit=train)
        df_processed = pd.DataFrame(X, columns= data.drop(['class'], axis=1).columns)
        df_processed['class'] = y
        df_processed.dropna(inplace=True)
        return df_processed

@click.command()
@click.argument('reductor')
def main(reductor):
    """
    Runs data processing scripts to turn raw data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing training data')
    # Load the data for training
    train_data = pd.read_csv("data/interim/train.csv")
    r = Reductor(reductor)
    data_processor = DataProcessor(r)
    # Process the training data 
    df_processed = data_processor.process_data(train_data,train=True)
    df_processed.to_csv("data/processed/train_processed.csv", index=False)

    # Reduce the dimension of the dataset (with fitting the reductor)
    df_reducted = data_processor.feature_reduction(df_processed,fit=True)
    df_reducted.to_csv("data/processed/train_reducted.csv", index=False)

    logger.info('Balancing training data with oversampling and undersampling methods')
    # Make different kind of dataset with oversampling and undersampling methods
    df_oversampled, df_undersampled = data_processor.balance_dataset(df_reducted)
    # Save the other two datasets
    df_oversampled.to_csv("data/processed/train_oversampled.csv", index=False)
    df_undersampled.to_csv("data/processed/train_undersampled.csv", index=False)

    logger.info('Processing testing data')
    # Load the data for testing
    test_data = pd.read_csv("data/interim/test.csv")
    # Process the testing data
    df_processed = data_processor.process_data(test_data,train=False)
    df_processed.to_csv("data/processed/test_processed.csv", index=False)
    # Reduce the dimension of the dataset (without fitting the reductor)
    df_reducted = data_processor.feature_reduction(df_processed,fit=False)
    df_reducted.to_csv("data/processed/test_reducted.csv", index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
 