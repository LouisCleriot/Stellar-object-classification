import click 
import logging
import pandas as pd

from src.models.KNNClassifier import KNNClassifier
from src.models.LogisticRegClassifier import LogisticRegClassifier
from src.models.NaiveBayesClassifier import NaiveBayesClassifier
from src.models.RandomForestClassifier import RandomForestClassifier
from src.models.SVMClassifier import SVMClassifier
from src.models.NNClassifier import NNClassifier

@click.command()
@click.option('-m','--model', type=click.Choice(['NaiveBayes', 'KNN', 'LogisticRegression', 'RandomForest', 'SVM', 'NeuralNetwork']), default='NaiveBayes')
@click.option('-d','--data', type=click.Choice(['scaled','scaled_umap', 'scaled_umap_oversampled', 'scaled_umap_undersampled']), default='scaled_umap')
@click.option('-n','--name', type=str, default=None)
def main(model, data, name):
    """ Trains model and saves it to models/ directory """
    logger = logging.getLogger(__name__)

    if model == 'NaiveBayes':
        clf = NaiveBayesClassifier()
    elif model == 'KNN':
        clf = KNNClassifier()
    elif model == 'LogisticRegression':
        clf = LogisticRegClassifier()
    elif model == 'RandomForest':
        clf = RandomForestClassifier()
    elif model == 'SVM':
        clf = SVMClassifier()
    elif model == 'NeuralNetwork':
        clf = NNClassifier()
    else:
        raise ValueError('Model name not recognized. Please choose from: NaiveBayes, KNN, LogisticRegression, RandomForest, SVM, NeuralNetwork')
    logger.info('Model: {}'.format(model))
    if data == 'scaled_umap':
        clf.name = f'{model}_scaled_umap'
        path = 'data/processed/train_reducted.csv'
    elif data == 'scaled':
        clf.name = f'{model}_scaled'
        path = 'data/processed/train_processed.csv'
    elif data == 'scaled_umap_oversampled':
        clf.name = f'{model}_scaled_umap_oversampled'
        path = 'data/processed/train_oversampled.csv'
    elif data == 'scaled_umap_undersampled':
        clf.name = f'{model}_scaled_umap_undersampled'
        path = 'data/processed/train_undersampled.csv'
        
    logger.info('Data: {}'.format(data))
    
    if name != None:
        clf.name = name
        
    logger.info('Model name: {}'.format(clf.name))
        
    # Read training data
    df_train = pd.read_csv(path)
    X_train,y_train = df_train.drop('class',axis=1),df_train['class']

    # Train the model 
    logger.info('Training model: {}'.format(clf.name))
    clf.train(X_train,y_train)
    logger.info('Model trained')
    # Save the model in models/ directory
    logger.info('Saving model: {} into /models directory'.format(clf.name))
    clf.save(path='models/')
    logger.info('Model saved')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()