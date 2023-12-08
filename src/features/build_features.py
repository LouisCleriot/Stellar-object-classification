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
            new_X = self.pca.fit(X[['u','g','z','r','i']])

        # Transform data
        new_X = self.pca.transform(X[['u','g','z','r','i']])

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
        sm = SMOTE(random_state=0)
        X_sm, y_sm = sm.fit_resample(X, y)
        df_oversampled = pd.DataFrame(X_sm, columns=X.columns)
        df_oversampled['class'] = y_sm
        spinner.succeed('data oversampled')

        spinner = Halo(text='undersampling the data', spinner='dots')
        spinner.start()
        rus = RandomUnderSampler(random_state=0)
        X_rus, y_rus = rus.fit_resample(X, y)
        df_undersampled = pd.DataFrame(X_rus, columns=X.columns)
        df_undersampled['class'] = y_rus
        spinner.succeed('data undersampled')

        return df_oversampled, df_undersampled
    
    def fit(self,data):
        """ Apply feature engineering on the data to fit the scaler and pca"""
        data = self.remove_correlation(data,fit=True)
        data = self.scale_data(data,fit=True)
        
    def process_test_data(self,data):
        """ Apply feature engineering on the test data"""
        data = self.remove_correlation(data,fit=False)
        data = self.scale_data(data,fit=False)
        return data
    

@click.command()
def main():
    """
    Runs data processing scripts to turn raw data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    
    data_processor = DataProcessor()
    
    # Load the data for training and testing
    spinner = Halo(text='Loading the data', spinner='dots').start()
    train_data_outlier = pd.read_csv("data/interim/train_without_outliers.csv")
    test_data_outlier = pd.read_csv("data/interim/test_without_outliers.csv")
    train_data = pd.read_csv("data/interim/train_with_outliers.csv")
    test_data = pd.read_csv("data/interim/test_with_outliers.csv")
    spinner.succeed('Data Loaded')
    
    # Save the train data (with and without outliers) for later use in /processed
    train_data.to_csv("data/processed/train_with_outliers.csv", index=False)
    train_data_outlier.to_csv("data/processed/train_without_outliers.csv", index=False)
    
    
    

    print("===========================================")
    print("Feature engineering on data ")
    
    spinner = Halo(text='Feature engineering on data with outliers', spinner='dots')
    # Fit the scaler and pca on the train data & apply the feature engineering on the test data
    data_processor.fit(train_data)
    test_data_processed = data_processor.process_test_data(test_data)
    # Save the processed test data
    test_data_processed.to_csv("data/processed/test_with_outliers.csv", index=False)
    spinner.succeed('Test data with outliers processed and saved')

    
    spinner = Halo(text='Feature engineering on data without outliers', spinner='dots')
    # Fit the scaler and pca on the train data & apply the feature engineering on the test data
    data_processor.fit(train_data_outlier)
    test_data_processed = data_processor.process_test_data(test_data_outlier)
    
    # Save the processed test data
    test_data_processed.to_csv("data/processed/test_without_outliers.csv", index=False)
    spinner.succeed('Test data without outliers processed and saved')
    print("Data processed")
    print("===========================================")
    
    
    
    
    print("===========================================")
    print('Balancing training data with oversampling and undersampling methods')
    # Make the data balanced for training dataset without outliers
    spinner = Halo(text='balancing datasets without outliers', spinner='dots')
    spinner.start()
    df_oversampled_outlier, df_undersampled_outlier = data_processor.balance_dataset(train_data_outlier)
    spinner.succeed('training data without outliers balanced')
    
    # Make the data balanced for training dataset with outliers
    spinner = Halo(text='balancing datasets with outliers', spinner='dots')
    spinner.start()
    df_oversampled, df_undersampled = data_processor.balance_dataset(train_data)
    spinner.succeed('training data with outliers balanced')
    
    # Save alls the balanced training dataset
    spinner = Halo(text='saving all training datasets', spinner='dots')
    spinner.start()
    df_oversampled.to_csv("data/processed/train_with_outliers_oversampled.csv", index=False)
    df_undersampled.to_csv("data/processed/train_with_outliers_undersampled.csv", index=False)
    df_oversampled_outlier.to_csv("data/processed/train_without_outliers_oversampled.csv", index=False)
    df_undersampled_outlier.to_csv("data/processed/train_without_outliers_undersampled.csv", index=False)
    spinner.succeed('all training datasets saved')
    print("Training data balanced")
    print("===========================================")
    
    
    
    
    print("===========================================")
    print("Processing test data with scaler and pca fitted on balanced training data")
    
    ## Oversampled dataset without outliers
    spinner = Halo(text='feature engineering of test data without outliers oversampled', spinner='dots').start()
    data_processor.fit(df_oversampled_outlier)
    test_data_oversampled_outlier = data_processor.remove_correlation(test_data_outlier,fit=False)
    test_data_oversampled_outlier.to_csv("data/processed/test_without_outliers_oversampled.csv", index=False)
    spinner.succeed('test data with outliers oversampled processed and saved')
    
    ## Undersampled dataset without outliers
    spinner = Halo(text='feature engineering of test data without outliers undersampled', spinner='dots')
    spinner.start()
    data_processor.fit(df_undersampled_outlier)
    test_data_undersampled_outlier = data_processor.remove_correlation(test_data_outlier,fit=False)
    test_data_undersampled_outlier.to_csv("data/processed/test_without_outliers_undersampled.csv", index=False)
    spinner.succeed('test data without outliers undersampled processed and saved')
    
    spinner = Halo(text='feature engineering of test data with outliers oversampled', spinner='dots')
    spinner.start()
    data_processor.fit(df_oversampled)
    test_data_oversampled = data_processor.process_test_data(test_data)
    test_data_oversampled.to_csv("data/processed/test_with_outliers_oversampled.csv", index=False)
    spinner.succeed('test data with outliers oversampled processed and saved')
    
    spinner = Halo(text='feature engineering of test data with outliers undersampled', spinner='dots')
    spinner.start()
    data_processor.fit(df_undersampled)
    test_data_undersampled = data_processor.process_test_data(test_data)
    test_data_undersampled.to_csv("data/processed/test_with_outlier_undersampled.csv", index=False)
    spinner.succeed('test data with outliers undersampled processed and saved')

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
 