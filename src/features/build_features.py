from sklearn.preprocessing import RobustScaler
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

        # Transform data
        new_X = self.pca.transform(X)

        # Add the reducted features to the new dataset
        df = pd.DataFrame(new_X, columns=['ugzri_1','ugzri_2','ugzri_3','ugzri_4','ugzri_5'])

        # Add original data 
        df['redshift'] = X['redshift']
        df['class'] = y
        return df
    
    def balance_dataset(self,data):
        X,y = self.split_data(data)

        spinner = Halo(text='oversampling the data', spinner='dots')
        spinner.start()
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X, y)
        df_oversampled = pd.DataFrame(X_sm, columns=X.columns)
        df_oversampled['class'] = y_sm
        spinner.succeed('data oversampled')

        spinner = Halo(text='undersampling the data', spinner='dots')
        spinner.start()
        cc = RandomUnderSampler(random_state=0)
        X_cc, y_cc = cc.fit_resample(X, y)
        df_undersampled = pd.DataFrame(X_cc, columns=X.columns)
        df_undersampled['class'] = y_cc
        spinner.succeed('data undersampled')

        return df_oversampled, df_undersampled
    

@click.command()
def main():
    """
    Runs data processing scripts to turn raw data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    print("===========================================")
    print("Processing data with scaling")

    # Load the data for training and testing
    spinner = Halo(text='Loading the data', spinner='dots')
    spinner.start()
    train_data_outlier = pd.read_csv("data/interim/train_without_outliers.csv")
    test_data_outlier = pd.read_csv("data/interim/test_without_outliers.csv")
    train_data = pd.read_csv("data/interim/train_with_outliers.csv")
    test_data = pd.read_csv("data/interim/test_with_outliers.csv")
    spinner.succeed('Data Loaded')
    train_data.to_csv("data/processed/train_with_outlier.csv", index=False)
    train_data_outlier.to_csv("data/processed/train_without_outlier.csv", index=False)
    
    data_processor = DataProcessor()
    
    print("===========================================")
    print('Balancing training data with oversampling and undersampling methods')
    # Make different kind of dataset with oversampling and undersampling methods
    spinner = Halo(text='balancing dataset without outliers', spinner='dots')
    spinner.start()
    df_oversampled_outlier, df_undersampled_outlier = data_processor.balance_dataset(train_data_outlier)
    spinner.succeed('training data without outliers balanced')
    spinner = Halo(text='balancing dataset with outliers', spinner='dots')
    spinner.start()
    df_oversampled, df_undersampled = data_processor.balance_dataset(train_data)
    spinner.succeed('training data with outliers balanced')
    spinner = Halo(text='saving all training dataset', spinner='dots')
    spinner.start()
    df_oversampled.to_csv("data/processed/train_with_outlier_oversample.csv", index=False)
    df_undersampled.to_csv("data/processed/train_with_outlier_undersampled.csv", index=False)
    df_oversampled_outlier.to_csv("data/processed/train_without_outlier_oversample.csv", index=False)
    df_undersampled_outlier.to_csv("data/processed/train_without_outlier_undersampled.csv", index=False)
    spinner.succeed('all training dataset saved')
    print("training data balanced")
    print("===========================================")
    
    
    print("===========================================")
    print("Feating scaling and PCA on test dataset")
    spinner = Halo(text='feature engineering of test data with outliers', spinner='dots')
    spinner.start()
    train_pca = data_processor.remove_correlation(train_data,fit=True)
    train_scaler = data_processor.scale_data(train_pca,fit=True)
    test_data_c = data_processor.remove_correlation(test_data,fit=False)
    test_data_c = data_processor.scale_data(test_data_c,fit=False)
    test_data_c.to_csv("data/processed/test_with_outlier.csv", index=False)
    spinner.succeed('test data with outliers processed and saved')
    
    spinner = Halo(text='feature engineering of test data without outliers', spinner='dots')
    spinner.start()
    train_pca_outlier = data_processor.remove_correlation(train_data_outlier,fit=True)
    train_scaler_outlier = data_processor.scale_data(train_pca_outlier,fit=True)
    test_data_outlier_c = data_processor.remove_correlation(test_data_outlier,fit=False)
    test_data_outlier_c = data_processor.scale_data(test_data_outlier_c,fit=False)
    test_data_outlier_c.to_csv("data/processed/test_without_outlier.csv", index=False)
    spinner.succeed('test data without outliers processed and saved')
    
    spinner = Halo(text='feature engineering of test data with outliers oversampled', spinner='dots')
    spinner.start()
    df_pca_oversampled_outlier = data_processor.remove_correlation(df_oversampled_outlier,fit=True)
    train_scaler_oversampled_outlier = data_processor.scale_data(df_pca_oversampled_outlier,fit=True)
    
    test_data_oversampled_outlier = data_processor.remove_correlation(test_data_outlier,fit=False)
    test_data_oversampled_outlier = data_processor.scale_data(test_data_oversampled_outlier,fit=False)
    test_data_oversampled_outlier.to_csv("data/processed/test_without_outlier_oversampled.csv", index=False)
    spinner.succeed('test data with outliers oversampled processed and saved')
    
    spinner = Halo(text='feature engineering of test data with outliers undersampled', spinner='dots')
    spinner.start()
    df_pca_undersampled_outlier = data_processor.remove_correlation(df_undersampled_outlier,fit=True)
    train_scaler_undersampled_outlier = data_processor.scale_data(df_pca_undersampled_outlier,fit=True)
    test_data_undersampled_outlier = data_processor.remove_correlation(test_data_outlier,fit=False)
    test_data_undersampled_outlier = data_processor.scale_data(test_data_undersampled_outlier,fit=False)
    test_data_undersampled_outlier.to_csv("data/processed/test_without_outlier_undersampled.csv", index=False)
    spinner.succeed('test data with outliers undersampled processed and saved')
    
    spinner = Halo(text='feature engineering of test data without outliers oversampled', spinner='dots')
    spinner.start()
    df_pca_oversampled = data_processor.remove_correlation(df_oversampled,fit=True)
    train_scaler_oversampled = data_processor.scale_data(df_pca_oversampled,fit=True)
    test_data_oversampled = data_processor.remove_correlation(test_data,fit=False)
    test_data_oversampled = data_processor.scale_data(test_data_oversampled,fit=False)
    test_data_oversampled.to_csv("data/processed/test_with_outlier_oversampled.csv", index=False)
    spinner.succeed('test data without outliers oversampled processed and saved')
    
    spinner = Halo(text='feature engineering of test data without outliers undersampled', spinner='dots')
    spinner.start()
    df_pca_undersampled = data_processor.remove_correlation(df_undersampled,fit=True)
    train_scaler_undersampled = data_processor.scale_data(df_pca_undersampled,fit=True)
    test_data_undersampled = data_processor.remove_correlation(test_data,fit=False)
    test_data_undersampled = data_processor.scale_data(test_data_undersampled,fit=False)
    test_data_undersampled.to_csv("data/processed/test_with_outlier_undersampled.csv", index=False)
    spinner.succeed('test data without outliers undersampled processed and saved')

    ## First dataset : only scaling
    #spinner = Halo(text='Scalling the data', spinner='dots')
    #spinner.start()
    # Process the training data 
    #train_scaled = data_processor.scale_data(train_data,fit=True)
    #train_scaled.to_csv("data/processed/train_scaled.csv", index=False)
    # Process the testing data
    #test_scaled = data_processor.scale_data(test_data,fit=False)
    #test_scaled.to_csv("data/processed/test_scaled.csv", index=False)
    #spinner.succeed('Data scaled')
    #print('scaled training data saved in data/processed/train_scaled.csv')
    ##print('scaled testing data saved in data/processed/test_scaled.csv')


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
 