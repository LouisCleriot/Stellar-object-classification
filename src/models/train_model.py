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
@click.argument('model', type=click.Choice(['NaiveBayes', 'KNN', 'LogisticRegression', 'RandomForest', 'SVM', 'NeuralNetwork']))
def main(model):
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
    
    # Read training data
    df_train = pd.read_csv("data/processed/train_reducted.csv")
    X_train,y_train = df_train.drop('class',axis=1),df_train['class']

    # Train the model 
    logger.info('Training model: {}'.format(model))
    clf.train(X_train,y_train)

    # Save the model in models/ directory
    logger.info('Saving model: {} into /models directory'.format(model))
    clf.save()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()