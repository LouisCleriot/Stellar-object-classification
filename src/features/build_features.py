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


    
def balance_dataset(data):
    X,y = data.drop('class', axis=1), data['class']

    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    df_oversampled = pd.DataFrame(X_sm, columns=X.columns)
    df_oversampled['class'] = y_sm


    cc = RandomUnderSampler(random_state=0)
    X_cc, y_cc = cc.fit_resample(X, y)
    df_undersampled = pd.DataFrame(X_cc, columns=X.columns)
    df_undersampled['class'] = y_cc

    return df_oversampled, df_undersampled
    

@click.command()
def main():
    """
    Runs data processing scripts to turn raw data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    ##################################################################
    #       Load and save the test datsets of data/interim           #
    ##################################################################
    spinner = Halo(text='Loading the test datasets of data/interim', spinner='dots')
    df_test_with_outliers = pd.read_csv('data/interim/test_with_outliers.csv')
    df_test_without_outliers = pd.read_csv('data/interim/test_without_outliers.csv')
    spinner.succeed('test datasets loaded')
    spinner = Halo(text="Saving the test datasets in data/processed/[folder_with/without]", spinner='dots')
    df_test_with_outliers.to_csv('data/processed/with_outliers/test.csv', index=False)
    df_test_without_outliers.to_csv('data/processed/without_outliers/test.csv', index=False)
    spinner.succeed('test datasets saved')
    
    ##################################################################
    #       Load and save the train datsets of data/interim          #
    ##################################################################
    spinner = Halo(text='Loading the train datasets of data/interim', spinner='dots')
    df_train_with_outliers = pd.read_csv('data/interim/train_with_outliers.csv')
    df_train_without_outliers = pd.read_csv('data/interim/train_without_outliers.csv')
    spinner.succeed('train datasets loaded')
    spinner = Halo(text="Saving the train datasets in data/processed/[folder_with/without]", spinner='dots')
    df_train_with_outliers.to_csv('data/processed/with_outliers/train.csv', index=False)
    df_train_without_outliers.to_csv('data/processed/without_outliers/train.csv', index=False)
    spinner.succeed('train datasets saved')
    
    ##################################################################
    #  Using the train datsets to do oversampling and undersampling  #
    ##################################################################
    spinner = Halo(text="balancing both datasets", spinner='dots')
    spinner.start()
    df_train_with_outliers_over, df_train_with_outliers_under = balance_dataset(df_train_with_outliers)
    df_train_without_outliers_over, df_train_without_outliers_under = balance_dataset(df_train_without_outliers)
    spinner.succeed('datasets balanced')
    
    ##################################################################
    #   Save the train balanced datsets in data/processed/[folder]   #
    ##################################################################
    spinner = Halo(text="Saving the balanced datasets in data/processed/[folder_with/without]", spinner='dots')
    df_train_with_outliers_over.to_csv('data/processed/with_outliers/train_over.csv', index=False)
    df_train_with_outliers_under.to_csv('data/processed/with_outliers/train_under.csv', index=False)
    df_train_without_outliers_over.to_csv('data/processed/without_outliers/train_over.csv', index=False)
    df_train_without_outliers_under.to_csv('data/processed/without_outliers/train_under.csv', index=False)
    spinner.succeed('balanced datasets saved')
    

if __name__ == '__main__':

    main()
 