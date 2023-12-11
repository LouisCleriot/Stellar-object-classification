from huggingface_hub import hf_hub_download
import joblib
import sys

REPO_ID = "Louzii/ift712"

dict_name = {
    "RF": "RandomForest_",
    "LR": "LogisticRegression_",
    "KNN": "knn_",
    "NB": "NaiveBayes_",
    "NN": "nn_",
    "SVM": "SVM_"}

list_outlier = ["with_outliers", "without_outliers"]
list_preprocess = ["_over", "_under", ""]
dir_path = "models/"
extension = ".joblib"

def download_model(model_name):
    for outlier in list_outlier:
        for preprocess in list_preprocess:
            model_file= model_name + '/' + dict_name[model_name]
            if preprocess != "":
                model_file = model_file + outlier + preprocess + extension
            else:
                model_file = model_file + outlier + extension
            try:
                hf_hub_download(repo_id=REPO_ID, filename=model_file,  local_dir=dir_path, local_dir_use_symlinks=False)
            except:
                print(f"{model_file} does not exist")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py [directory_name]")
        print("possible directory_name: RF, LR, KNN, NB, NN, SVM or ALL (to download all models)")
        print("Example: make download_model MODEL=RF")
    else:
        model_name = sys.argv[1]
        if model_name in dict_name.keys():
            download_model(model_name)
        elif model_name == "ALL":
            for model_name in dict_name.keys():
                download_model(model_name)
        else:
            print("Usage: python script.py [directory_name]")
            print("possible directory_name: RF, LR, KNN, NB, NN, SVM or ALL (to download all models)")
            print("Example: make download_model MODEL=RF")
        