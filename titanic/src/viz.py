import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd

sns.set(style="darkgrid", color_codes=True)

titanic = pd.read_csv('data/train.csv')

titanic['Fare_sqrt'] = titanic.apply(lambda row: sqrt(row['Fare']), axis=1)

g = sns.jointplot("Age", "Fare_sqrt", data=titanic, kind="reg",
                  xlim=(0, 70), ylim=(0, 20), color="r", size=7)

plt.show()
