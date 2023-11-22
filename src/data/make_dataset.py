# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    data = pd.read_csv(input_filepath+"/star_classification.csv")
    #get rid of not use properties
    data=data.drop(['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','plate','MJD','fiber_ID'], axis=1)
    #get rid of the outlier of u,g,z
    data = data[data['g'] > -2000]
    data = data[data['z'] > -2000]
    data = data[data['u'] > -2000]
    #scale the data
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(data.drop(['class'], axis=1))
    df_scaled = pd.DataFrame(df_scaled, columns=data.columns[:-1])
    df_scaled['class'] = data['class']
    #save the data
    df_scaled.to_csv(output_filepath+"/star_classification.csv", index=False)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
