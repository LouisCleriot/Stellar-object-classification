import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import click
import umap


def plot_roc_curve (model, X_test, y_test):
    y_score = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    sns.set()
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='red',
            lw=lw, label='Class 0 (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='green',
                lw=lw, label='Class 1 (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='blue',
                lw=lw, label='Class 2 (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    # auc score
    print("auc score for class 0: ", roc_auc[0])
    print("auc score for class 1: ", roc_auc[1])
    print("auc score for class 2: ", roc_auc[2])


def display_umap2D(data, labels, predicted_labels,title):
    """ Display the data in 2D with umap. The first plot colored the data with true labels, the second with the predicted labels """
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(data)
    data['umap_0'] = embedding[:, 0]
    data['umap_1'] = embedding[:, 1]

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(data['umap_0'], data['umap_1'], c=labels, cmap='Spectral', s=10)
    axes[1].scatter(data['umap_0'], data['umap_1'], c=predicted_labels, cmap='Spectral', s=10)
    axes[0].set_title('True labels')
    axes[1].set_title('Predicted labels')

    fig.suptitle(title)
    plt.show()
    

def check_model_exists(ctx, param, value):
    """ Check if model exists in models/ directory """
    if value == None:
        model = ctx.params.get('model')
        data = ctx.params.get('data')
        value = f'{model}_{data}'
    if not os.path.exists(f'models/{value}.joblib'):
        raise click.BadParameter(f"The file {value}.joblib does not exist in models/ directory. \nPlease train the model first or choose another trained model in the /models directory.")
    return value

