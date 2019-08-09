import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "Churn_Modelling.csv"

df = pd.read_csv(PATH)

df['Gender'] = df['Gender'].replace(["Female"], 0)
df['Gender'] = df['Gender'].replace(["Male"], 1)

for i,x in enumerate(pd.unique(df['Geography'])):
    print x
    df['Geography'] = df['Geography'].replace([x], i)

df.head()

# VISUALISATION
neg = df.loc[df['Exited'] == 0]
pos = df.loc[df['Exited'] == 1]

pltt = neg['Age'].plot.hist()
pltt.plot()
pltt = pos['Age'].plot.hist()
pltt.plot()
plt.show()

pltt = neg['NumOfProducts'].plot.hist()
pltt.plot()
pltt = pos['NumOfProducts'].plot.hist()
pltt.plot()
plt.show()

pltt = neg['IsActiveMember'].plot.hist()
pltt.plot()
pltt = pos['IsActiveMember'].plot.hist()
pltt.plot()
plt.show()

pltt = neg['Gender'].plot.hist()
pltt.plot()
pltt = pos['Gender'].plot.hist()
pltt.plot()
plt.show()

pltt = neg['Geography'].plot.hist()
pltt.plot()
pltt = pos['Geography'].plot.hist()
pltt.plot()
plt.show()



