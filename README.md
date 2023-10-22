projet-ift712
==============================

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

Repo git organisation
------------

### Le repo git suit l'organisation Git Flow. 

1 branche master \
1 branche develop \
1 branche feature par fonctionnalité \
    ex : feature/random-forest \
         feature/data-exploration  \
1 branche hotfix par correction de bug 

### comment utiliser les branches

La branche principale de développement est la branche develop.  \
Pour chaque nouvelle fonctionnalité une branches features est créées à partir de develop et mergées dans develop une fois terminée.  \
On ne push jamais sur develop directement.  \
On ne push jamais sur master directement.  \
On merge develop dans master à chaque fin de sprint. Quand un livrable fonctionne sans bug \
En cas de bug sur master, on crée une branche hotfix à partir de master, on corrige le bug et on merge dans master et develop. \

### commandes git

verifier la branch courante
```bash
git branch
```
creer une nouvelle branche
```bash
# de zero
git checkout -b <branch_name>
# a partir d'une branche existante (ex: creer une branche feature/random-forest a partir de develop)
git checkout develop
git checkout -b feature/random-forest
```

changer de branche
```bash
git checkout <branch_name>
```
merger une branche dans une autre (ex: merge feature/random-forest dans develop)
```bash
git checkout feature/random-forest
git pull origin develop
git checkout develop
git merge feature/random-forest
git push origin develop
#supprimer la branche localement
git branch -d feature-branch 
#supprimer la branche sur le remote
git push origin --delete feature-branch 
```
commiter sur une branche
```bash
git add .
git commit -m "message"
git push origin <branch_name>
```





Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Initialisation du projet
------------


### Docker

1. Installer docker

verifier que docker est installé
```bash
chmod +x ./checkDocker.sh
./checkDocker.sh
```
Si ce n'est pas le cas, installer docker grace au commandes suivante
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
2. Lancer le container

```bash
chmod +x ./buildandrunDocker.sh
./buildandrunDocker.sh
```

### Virtualenv

0. Installer python 11.6

```bash
sudo apt-get install python3.11
```

1. Installer virtualenv

```bash
sudo apt install python3.11-venv
```

2. Créer un environnement virtuel

```bash
python3.11 -m venv env
```

3. Activer l'environnement virtuel

```bash
source env/bin/activate
```

4. Installer les dépendances

```bash
pip install -r requirements.txt
```





--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
