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




class DataProcessor :

    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=5)
        
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
            new_X = self.pca.fit(X)
        new_X = self.pca.transform(X)

        # Add the reducted features to the new dataset
        df = pd.DataFrame(new_X, columns=['ugzri_1','ugzri_2','ugzri_3','ugzri_4','ugzri_5'])

        # Add original data 
        df['redshift'] = X['redshift']
        df['class'] = y
        return df
    
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
    

@click.command()
def main():
    """
    Runs data processing scripts to turn raw data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # Load the data for training and testing
    train_data = pd.read_csv("data/interim/train_without_outliers.csv")
    test_data = pd.read_csv("data/interim/test_without_outliers.csv")

    data_processor = DataProcessor()

    ## First dataset : only scaling
    logger.info('Making first datasets : only scaling')
    # Process the training data 
    train_scaled = data_processor.scale_data(train_data,fit=True)
    train_scaled.to_csv("data/processed/train_scaled.csv", index=False)
    # Process the testing data
    test_scaled = data_processor.scale_data(test_data,fit=False)
    test_scaled.to_csv("data/processed/test_scaled.csv", index=False)

    ## Second dataset : scaling + removing correlation
    logger.info('Making second datasets : scaling + removing correlation')
    # Apply PCA on u,g,z,r,i of train data (with fitting the reductor)
    train_pca = data_processor.remove_correlation(train_data,fit=True)
    train_pca = data_processor.scale_data(train_pca,fit=True)
    train_pca.to_csv("data/processed/train_scaled_pca.csv", index=False)
    # Apply PCA on test data (without fitting the reductor)
    test_pca = data_processor.remove_correlation(test_data,fit=False)
    test_pca = data_processor.scale_data(test_pca,fit=False)
    test_pca.to_csv("data/processed/test_scaled_pca.csv", index=False)

    ## Third and Fourth dataset : scaling + removing correlation + SMOTE(oversampling) / ClusterCentroids(undersampling)
    logger.info('Making third and fourth datasets : scaling + removing correlation + SMOTE(oversampling) / ClusterCentroids(undersampling)')
    # Make different kind of dataset with oversampling and undersampling methods
    df_oversampled, df_undersampled = data_processor.balance_dataset(train_pca)
    # Save the other two datasets
    df_oversampled.to_csv("data/processed/train_scaled_pca_oversampled.csv", index=False)
    df_undersampled.to_csv("data/processed/train_scaled_pca_undersampled.csv", index=False)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
 