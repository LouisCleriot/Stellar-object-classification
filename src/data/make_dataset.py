# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF
import umap


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making train and test data sets from raw data')
    data = pd.read_csv(input_filepath+"/star_classification.csv")
    #feature selection
    data = data.drop(['obj_ID','run_ID','rerun_ID','cam_col','field_ID','plate','MJD','fiber_ID','spec_obj_ID','alpha','delta'], axis=1)
    # Detection of outliers
    reducer = umap.UMAP(set_op_mix_ratio=0.008).fit(data.drop('class', axis=1))
    outliers_score = LOF(contamination=0.1).fit_predict(reducer.embedding_)
    # Remove outliers
    data = data[outliers_score == 1]
    data = data.reset_index(drop=True)
    # Split the data for training and testing
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    logger.info(f'Saving the data in {output_filepath} folder')
    # Save the data
    train.to_csv(output_filepath+"/train.csv", index=False)
    test.to_csv(output_filepath+"/test.csv", index=False)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
