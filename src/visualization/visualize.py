import click
import logging
from src.helper import visualize_results
import pandas as pd
from src.models.Classifier import Classifier
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.option('-m','--model', type=click.Choice(['NaiveBayes', 'KNN', 'LogisticRegression', 'RandomForest', 'SVM', 'NeuralNetwork']), default='NaiveBayes')
@click.option('-d','--data', type=click.Choice(['scaled','scaled_umap']), default='scaled_umap')
@click.option('-n','--name', type=str, default='NaiveBayes_scaled_umap')
def main(model, data, name):
    """ Trains model and saves it to models/ directory """
    logger = logging.getLogger(__name__)

    # load prediction data
    prediction_data = pd.read_csv(f'data/predictions/labels_{name}.csv')
    y_pred = prediction_data['class']

    # load test data
    test_data = pd.read_csv(f'data/processed/test_reducted.csv')
    # split data and labels
    X_test,y_test = test_data.drop('class',axis=1),test_data['class']

    # evaluate model
    logger.info('Evaluating model: {}'.format(name))
    clf = Classifier()
    clf.load(new_name=name,path='models/')

    # visualize results
    logger.info('Visualizing results')

    label_encoder = LabelEncoder().fit(y_test)
    y_test = label_encoder.transform(y_test)
    y_pred = label_encoder.transform(y_pred)

    fig,ax = plt.subplots(2,2)
    axes = ax.flatten()

    # umap 2D
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(X_test)

    axes[0].scatter(embedding[:, 0], embedding[:, 1], c=y_test, cmap='Spectral', s=10)
    axes[1].scatter(embedding[:, 0], embedding[:, 1], c=y_pred, cmap='Spectral', s=10)
    
    axes[0].set_title('True labels')
    axes[1].set_title('Predicted labels')

    #conf matrix

    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, ax=axes[2])
    
    #roc curve
    plt.show()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()