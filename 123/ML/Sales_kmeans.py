import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

%matplotlib inline

df = pd.read_csv("/content/sales_data_sample.csv", encoding='latin1')

df

list = ["PRICEEACH", "SALES"]

data = df[list]

data

df.columns

K=range(1, 7)
wss = []

wss = [] # Reinitialize wss here to ensure it's empty before appending new values
for k in K:
 kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
 kmeans=kmeans.fit(data)
 wss_iter = kmeans.inertia_ 
 wss.append(wss_iter)

mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})

mycenters

sns.scatterplot(x = 'Clusters', y = 'WSS', data = mycenters)