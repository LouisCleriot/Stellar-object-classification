# Notebooks

This directory contains the notebooks used to generate the results in the paper. The notebooks are organized as follows:

- `Data` contains the notebooks used to explore the data and see the choices made for the features or the way data is processed.
- `Models` contains the notebooks used to train the models and see the choices made for the hyperparameters. We also use this notebooks to see the results of the models and compare them.

## Data

- `DataExploration.ipynb` contains the basic exploration of the data and preliminary outiers detection.
- `DataCorrelation.ipynb` contains the reflexion on the best way to decorrelate the data, the choice between PCA, UMAP and t-SNE and the choice of the number of components.
- `SamplingMethods.ipynb` contains the reflexion on the best way to sample the data, for oversampling and undersampling.
- `DataProjection.ipynb` contains the reflexion on the best way to project the data, the choice between PCA, UMAP, t-SNE and LDA to visualize the data it in 2D and 3D.

## Models
- `[model_name]-training.ipynb` contains the training of the model `[model_name]` and the choice of the hyperparameters, with the possibility to choose the dataset and save the trained model.
- `comparison.ipynb` contains the comparison of the different types of models, the figure can be generated directly with `make visualize`.
- `choosing_model.ipynb` compare the models performance on diverse preprocessing methods and datasets, and choose the best model for each dataset.