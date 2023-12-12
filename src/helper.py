import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder


def plot_roc_curve(model, X_test, y_test):
    le = LabelEncoder()
    y_test = le.fit(y_test).transform(y_test)
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
    plt.plot(
        fpr[0],
        tpr[0],
        color="red",
        lw=lw,
        label="Class 0 (area = %0.2f)" %
        roc_auc[0])
    plt.plot(
        fpr[1],
        tpr[1],
        color="green",
        lw=lw,
        label="Class 1 (area = %0.2f)" % roc_auc[1],
    )
    plt.plot(
        fpr[2],
        tpr[2],
        color="blue",
        lw=lw,
        label="Class 2 (area = %0.2f)" %
        roc_auc[2])
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    # auc score
    print("auc score for class 0: ", roc_auc[0])
    print("auc score for class 1: ", roc_auc[1])
    print("auc score for class 2: ", roc_auc[2])
