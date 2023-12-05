# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF
import umap
from tqdm import tqdm
from halo import Halo

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making train and test data sets from raw data')

    spinner = Halo(text='Loading Dataset', spinner='pipe')
    spinner.start()
    data = pd.read_csv(input_filepath + "/star_classification.csv")
    data = data.drop(['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'plate', 'MJD', 'fiber_ID', 'spec_obj_ID', 'alpha', 'delta'], axis=1)
    spinner.succeed('Dataset Loaded')

    spinner = Halo(text='Detection of outliers', spinner='pipe')
    spinner.start()
    reducer = umap.UMAP(set_op_mix_ratio=0.008).fit(data.drop('class', axis=1))
    outliers_score = LOF(contamination=0.1).fit_predict(reducer.embedding_)
    spinner.succeed('Outliers Detected')

    spinner = Halo(text='Removing outliers', spinner='pipe')
    spinner.start()
    data = data[outliers_score == 1]
    data = data.reset_index(drop=True)
    spinner.succeed('Outliers Removed')

    spinner = Halo(text='Splitting and saving the data', spinner='pipe')
    spinner.start()
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train.to_csv(output_filepath + "/train.csv", index=False)
    test.to_csv(output_filepath + "/test.csv", index=False)
    spinner.succeed('Data Split and Saved')
    logger.info('preprocessed data saved in %s', output_filepath + "/train.csv and " + output_filepath + "/test.csv")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
