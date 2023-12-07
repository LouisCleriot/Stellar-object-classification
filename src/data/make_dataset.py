# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope as EE
import umap
from tqdm import tqdm
from halo import Halo

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        NOTE: For more details about the preprocessing of the data, please refer to the notebook in notebooks/DataVisualization.ipynb
    """
    logger = logging.getLogger(__name__)

    spinner = Halo(text='Loading Dataset', spinner='pipe')
    spinner.start()
    data = pd.read_csv(input_filepath+"/star_classification.csv")
    spinner.succeed('Dataset Loaded')

    spinner = Halo(text='Cleaning dataset', spinner='pipe')
    spinner.start()
    # Remove duplicate
    data.drop_duplicates(subset='obj_ID',keep='first')

    # Remove impossible samples (no galaxy can have a 0 redshift)
    data.drop((data[(data["class"] == "GALAXY") & (data["redshift"] == 0)].index))
    data = data[data['g'] > -2000]
    data = data[data['z'] > -2000]
    data = data[data['u'] > -2000]

    # Feature selection
    data = data.drop(['obj_ID','run_ID','rerun_ID','cam_col','field_ID','plate','MJD','fiber_ID','spec_obj_ID','alpha','delta'], axis=1)
    spinner.succeed('Dataset Cleaned')

    # Save train and test dataset before removing outliers
    spinner = Halo(text='Splitting and saving the data', spinner='pipe')
    spinner.start()
    # Split the data for training and testing with stratify sampling to keep the same proportion of classes in train and test
    train, test = train_test_split(data, test_size=0.2, random_state=0, stratify=data['class'])
    spinner.succeed('Data Split and Saved')

    # Save the data
    train.to_csv(output_filepath+"/train_with_outliers.csv", index=False)
    test.to_csv(output_filepath+"/test_with_outliers.csv", index=False)

    # Detection of outliers
    spinner = Halo(text='Detection of outliers', spinner='pipe')
    spinner.start()
    reducer = umap.UMAP(set_op_mix_ratio=0.008).fit(data.drop('class', axis=1))
    outliers_score = EE(contamination=0.04).fit_predict(reducer.embedding_)
    spinner.succeed('Outliers Detected')

    # Remove outliers
    spinner = Halo(text='Removing outliers', spinner='pipe')
    spinner.start()
    data = data[outliers_score == 1]
    data = data.reset_index(drop=True)
    spinner.succeed('Outliers Removed')

    # Split the data for training and testing
    spinner = Halo(text='Splitting and saving the data', spinner='pipe')
    spinner.start()
    train, test = train_test_split(data, test_size=0.2, random_state=0, stratify=data['class'])
    
    # Save the data
    train.to_csv(output_filepath+"/train_without_outliers.csv", index=False)
    test.to_csv(output_filepath+"/test_without_outliers.csv", index=False)
    spinner.succeed('Data Split and Saved')
   

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
