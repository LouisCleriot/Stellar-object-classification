# projet-ift712

Les étudiantes et étudiants inscrits au cours IFT712 sont tenus de faire un projet de session en python en équipe de 2 ou 3 (obligatoire). Le projet a pour objectif de tester au moins six méthodes de classification sur une base de données Kaggle (www.kaggle.com) avec la bibliothèque scikit-learn (https://scikit-learn.org). Les équipes sont libres de choisir la base de données de leur choix, mais une option simple est celle du challenge de classification de feuilles d’arbres (www.kaggle.com/c/leaf-classification). Pour ce projet, on s’attend à ce que les bonnes pratiques de cross-validation et de recherche d’hyper-paramètres soient mises de l’avant pour identifier la meilleure solution possible pour résoudre le problème. 

Le barême de correction est le suivant : \
Qualité du code - Commentaires 	 /10 \
Choix de design 	 /10 \
Gestion de projet (Git) 	 /10 \
Rapport 	 /70 \
... Démarche scientifique \
	 /50 \
... Analyse des résultats \
	 /20


Données choisies
------------
Stellar Classification Dataset - SDSS17

En astronomie, la classification stellaire est la classification des étoiles sur la base de leurs caractéristiques spectrales. Le système de classification des galaxies, des quasars et des étoiles est l'un des plus fondamentaux de l'astronomie. Le catalogage précoce des étoiles et de leur distribution dans le ciel a permis de comprendre qu'elles constituent notre propre galaxie et, suite à la distinction qu'Andromède était une galaxie séparée de la nôtre, de nombreuses galaxies ont commencé à être étudiées grâce à la construction de télescopes plus puissants. Ce datasat vise à classer les étoiles, les galaxies et les quasars en fonction de leurs caractéristiques spectrales.

https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data


Repo git organisation
------------

### Le repo git suit l'organisation Git Flow. 

1 branche master \
1 branche develop \
1 branche feature par fonctionnalité \
    ex : feature/random-forest \
         feature/data-exploration  \


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




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
