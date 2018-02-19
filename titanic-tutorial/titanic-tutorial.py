import matplotlib.pyplot as plt
# %matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
from tensorflow.contrib import bayesflow

titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])

# Check out a sampling of the table
titanic_df.head()

# See what the real survival rate was
titanic_df['survived'].mean()

# See what survival rate (and other factors) were by class
titanic_df.groupby('pclass').mean()

# Group by class and then sex
class_sex_grouping = titanic_df.groupby(['pclass', 'sex']).mean()

# Graph those results
class_sex_grouping['survived'].plot.bar()
plt.tight_layout()
plt.show()

# Graph by age
group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()
plt.tight_layout()
plt.show()

# Check out our missing values
print(titanic_df.count())