import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('data/raw/star_classification.csv')

# check data for missing values
print("number of misssing variable : \n",df.isnull().sum())
# check data for duplicates
df.duplicated().sum()
print("number of duplicates variable : ",df.duplicated().sum())




