import click 
import logging
import pandas as pd

from src.models.KNNClassifier import KNNClassifier
from src.models.LogisticRegClassifier import LogisticRegClassifier
from src.models.NaiveBayesClassifier import NaiveBayesClassifier
from src.models.RandomForestClassifier import RandomForestClassifier
from src.models.SVMClassifier import SVMClassifier
from src.models.NNClassifier import NNClassifier
from src.helper import check_model_exists

@click.command()
@click.option('-m','--model', type=click.Choice(['NaiveBayes', 'KNN', 'LogisticRegression', 'RandomForest', 'SVM', 'NeuralNetwork']), default='NaiveBayes')
@click.option('-d','--data', type=click.Choice(['scaled','scaled_umap']),default='scaled_umap')
@click.option('-n','--model_name', type=str, default=None, callback=check_model_exists)
def main(model,model_name,data):
    """ Load model {model_name}.joblib from models/ directory and predict labels of the data """
    logger = logging.getLogger(__name__)

    if model_name == None:
        model_name = model+'_'+data

    if model == 'NaiveBayes':
        clf = NaiveBayesClassifier(name=model_name)
    elif model == 'KNN':
        clf = KNNClassifier(name=model_name)
    elif model == 'LogisticRegression':
        clf = LogisticRegClassifier(name=model_name)
    elif model == 'RandomForest':
        clf = RandomForestClassifier(name=model_name)
    elif model == 'SVM':
        clf = SVMClassifier(name=model_name)
    elif model == 'NeuralNetwork':
        clf = NNClassifier(name=model_name)
    else:
        raise ValueError('Model name not recognized. Please choose from: NaiveBayes, KNN, LogisticRegression, RandomForest, SVM, NeuralNetwork')

    # Choose the data to predict
    if data == 'scaled_umap':
        path = 'data/processed/test_reducted.csv'
    elif data == 'scaled':
        path = 'data/processed/test_processed.csv'

    # Load test data
    df_test = pd.read_csv(path)
    # Split data and labels
    X_test,y_test = df_test.drop('class',axis=1),df_test['class']

        
    # Load model
    logger.info("Loading model")
    clf.load(path='models/')

    # Predict
    logger.info("Prediction")
    labels = pd.DataFrame(clf.predict(X_test),columns=['class'])
    labels.to_csv(f"data/predictions/labels_{model_name}.csv",index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()