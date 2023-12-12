import pandas as pd
from src.models.RandomForestClassifier import RFClassifier
from src.models.KNNClassifier import KNNClassifier
from src.models.SVMClassifier import SVMClassifier
from src.models.NaiveBayesClassifier import NaiveBayesClassifier
from src.models.LogisticRegClassifier import LogisticRegClassifier
from src.models.NNClassifier import NNClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sns
import matplotlib.pyplot as plt
from halo import Halo

sns.set_style("darkgrid")
sns.set_context("paper")


if __name__ == "__main__":
    # check if data is present
    spinner = Halo(text="Loading data", spinner="dots")
    try:
        test_data = pd.read_csv("data/processed/with_outliers/test.csv")
        y_train = pd.read_csv(
            "data/processed/with_outliers/train.csv")["class"]
        spinner.succeed("Data loaded successfully")
    except BaseException:
        spinner.fail("Loading data failed")
        print(
            "Data not present, please make sure you have preprocessed data,"
            "if not enter in command line:"
        )
        print("make data")
        print("make feautures")
        exit(0)
    # Split into X and y
    y_test = test_data["class"]
    X_test = test_data.drop("class", axis=1)
    # Encode labels
    le = LabelEncoder().fit(y_train)
    y_test = le.transform(y_test)
    spinner = Halo(text="Loading models", spinner="dots")
    try:
        RF = RFClassifier()
        RF.load(new_name="RandomForest_with_outliers", path="models/RF/")
        LR = LogisticRegClassifier()
        LR.load(new_name="LogisticRegression_with_outliers", path="models/LR/")
        NB = NaiveBayesClassifier()
        NB.load(new_name="NaiveBayes_with_outliers", path="models/NB/")
        SVM = SVMClassifier()
        SVM.load(new_name="SVM_with_outliers", path="models/SVM/")
        KNN = KNNClassifier()
        KNN.load(new_name="knn_with_outliers", path="models/KNN/")
        NN = NNClassifier()
        NN.load(new_name="nn_with_outliers", path="models/NN/")
        spinner.succeed("Models loaded successfully")
    except BaseException:
        spinner.fail("Loading models failed")
        print(
            "Models not present, please make sure you have trained models"
            "or download them, if not enter in command line:"
        )
        print("make download_models")
        exit(0)
    models = [RF, LR, NB, SVM, KNN, NN]
    reports = pd.DataFrame(
        columns=[
            "model",
            "inference Time (ms)",
            "accuracy",
            "f1_macro",
            "f1_GALAXY",
            "f1_QSO",
            "f1_STAR",
        ]
    )
    for model in models:
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report = {}
        report["inference Time (ms)"] = (end_time - start_time) * 1000
        report["model"] = model.name
        report["accuracy"] = class_report["accuracy"]
        report["f1_macro"] = class_report["macro avg"]["f1-score"]
        for i in range(len(le.classes_)):
            report[f"f1_{le.inverse_transform([i])[0]}"] = class_report[f"{i}"][
                "f1-score"
            ]
        reports = pd.concat(
            [reports, pd.DataFrame([report])], ignore_index=True)

    # comparison metrics
    spinner = Halo(text="Plotting metrics", spinner="dots")
    spinner.start()
    reshaped_reports = reports.melt(
        id_vars=["model"],
        value_vars=reports.columns[2:],
        var_name="metric",
        value_name="score",
    )
    ax = sns.barplot(data=reshaped_reports, x="metric", y="score", hue="model")
    ax.set(ylim=(0.8, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Comparison of main metrics for different models")
    ax.figure.savefig(
        "reports/figures/comparison_metrics.png",
        bbox_inches="tight")
    spinner.succeed("Metrics plotted successfully")
    plt.show()

    # inference time
    spinner = Halo(text="Plotting inference time", spinner="dots")
    spinner.start()
    ylim = 100
    ax = sns.barplot(
        data=reports,
        y="inference Time (ms)",
        hue="model",
    )
    ax.set_title("Comparison of inference time for different models")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set(ylim=(0, ylim))
    for p in ax.patches:
        if p.get_height() == 0:
            continue
        elif p.get_height() > ylim:
            ax.annotate(
                f"{p.get_height():.2f}",
                # Position in the middle of the bar
                (p.get_x() + p.get_width() / 2.0, ylim - 10),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    ax.figure.savefig(
        "reports/figures/comparison_inference_time.png", bbox_inches="tight"
    )
    spinner.succeed("Inference time plotted successfully")
    plt.show()

    # plot roc curve
    spinner = Halo(text="Plotting roc curve", spinner="dots")
    spinner.start()
    from sklearn.metrics import roc_curve, auc

    n_classes = 3

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for model in models:
        y_prob = model.model.predict_proba(X_test)
        for i, ax in enumerate([ax1, ax2, ax3]):
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model.name} (area = {roc_auc:0.3f})")

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.plot([0, 1], [0, 1], "k--", label="Baseline")
        ax.set_xlim([-0.02, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            "ROC curve for class {}".format(
                le.inverse_transform(
                    [i])[0]))
        ax.legend(loc="lower right")
        ax.grid()
    fig1.savefig("reports/figures/roc_curve_galaxy.png", bbox_inches="tight")
    fig2.savefig("reports/figures/roc_curve_qso.png", bbox_inches="tight")
    fig3.savefig("reports/figures/roc_curve_star.png", bbox_inches="tight")
    spinner.succeed("Roc curve plotted successfully")
    plt.show()

    # plot precision recall curve
    spinner = Halo(text="Plotting precision recall curve", spinner="dots")
    spinner.start()
    from sklearn.metrics import precision_recall_curve, auc

    n_classes = 3

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for model in models:
        y_prob = model.model.predict_proba(X_test)
        for i, ax in enumerate([ax1, ax2, ax3]):
            precision, recall, _ = precision_recall_curve(
                y_test, y_prob[:, i], pos_label=i
            )

            pr_auc = auc(recall, precision)
            ax.plot(
                recall,
                precision,
                label=f"{model.name} (area = {pr_auc:0.3f})")

    for i, ax in enumerate([ax1, ax2, ax3]):
        baseline = len(y_test[y_test == i]) / len(y_test)
        ax.plot([0, 1], [baseline, baseline], "k--", label="Baseline")
        ax.set_xlim([-0.02, 1.05])
        ax.set_ylim([baseline - 0.02, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            "Precision-Recall curve for class {}".format(le.inverse_transform([i])[0])
        )
        ax.legend(loc="best")

    fig1.savefig(
        "reports/figures/precision_recall_curve_galaxy.png",
        bbox_inches="tight")
    fig2.savefig(
        "reports/figures/precision_recall_curve_qso.png",
        bbox_inches="tight")
    fig3.savefig(
        "reports/figures/precision_recall_curve_star.png",
        bbox_inches="tight")
    spinner.succeed("Precision recall curve plotted successfully")
    plt.show()
    print("all the plots are saved in reports/figures/")
