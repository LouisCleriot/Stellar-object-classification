# Stellar object classification

Students enrolled in the IFT712 course are required to complete a team python project in groups of 2 or 3 (mandatory). The project aims to test at least six classification methods on a Kaggle database (www.kaggle.com) using the scikit-learn library (https://scikit-learn.org). Teams are free to choose the database of their choice, but a simple option is the tree leaf classification challenge (www.kaggle.com/c/leaf-classification). For this project, we expect good practices of cross-validation and hyper-parameter tuning to be employed to identify the best possible solution for solving the problem.

The grading scale is as follows: \
Code Quality - Comments /10 \
Design Choices /10 \
Project Management (Git) /10 \
Report /70 \
... Scientific Approach \
/50  \
... Analysis of Results \
/20 \

Selected Data
------------
Stellar Classification Dataset - SDSS17

In astronomy, stellar classification is the classification of stars based on their spectral characteristics. The classification system for galaxies, quasars, and stars is one of the most fundamental in astronomy. Early cataloging of stars and their distribution in the sky has helped understand that they constitute our own galaxy, and following the distinction that Andromeda was a separate galaxy from ours, many galaxies began to be studied thanks to the construction of more powerful telescopes. This dataset aims to classify stars, galaxies, and quasars based on their spectral characteristics.

https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data


Git repo organization
------------

### The git repo follows the Git Flow organization.

1 master branch \
1 develop branch \
1 feature branch per feature \
e.g., feature/random-forest \
feature/data-exploration \


Project Organization
------------

    ├── LICENSE            <- MIT License
    ├── Makefile           <- Makefile with commands like `make data` or `make features`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models
    │   
    ├── notebooks          <- Jupyter notebooks (for training and data exploration)
    │   ├── Data           <- Data exploration notebooks
    │   └── Models         <- Training notebooks
    │
    ├── reports           
    │   └── figures        <- Generated graphics and figures with make visualize
    |   └── report.pdf
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   |── make_dataset.py
    |   |   └── download.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Class for each model                
    │   │   ├── Classifiers.py
    │   │   └── download.py
    │   │
    │   └── visualization  <- Scripts to create comparative plots of the models
    │       └── visualize.py

## Initialize project


### Docker

1. Install docker

If docker is not installed on your machine, you can install it with the following commands:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null    
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
2. Build and run docker

```bash
chmod +x ./buildandrunDocker.sh
./buildandrunDocker.sh
```

### Virtualenv

0. Install python 3.9

```bash
sudo apt-get install python3.9
```

1. Use makefile to create virtualenv

```bash
make create_environment
```
2. Install requirments

```bash
make requirements
```

### How to use the project

1. Download the data with the link at the start of the README.md or :

```bash
make download_data
```

2. Create intermediate data

```bash
make data
```

3. Create features for final data

```bash
make features
```
4. Train the models with the notebooks in the notebooks/Models folder or download the models :

```bash
make download_models
```
5. Visualize the results with the notebooks in the notebooks/Data folder or download the results :

```bash
make visualize
```

All this steps can be done with the following command :

```bash
make run
```
### usefull command

```bash
./launchnotebook.sh 
```
is used to launch Jupiter daemon to use notebooks on web navigator




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
