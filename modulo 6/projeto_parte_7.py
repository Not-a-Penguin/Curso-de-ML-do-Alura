import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# importando os dados e criando o dataframe
uri = "https://raw.githubusercontent.com/alura-cursos/alura-clustering-validation/base-de-dados/CC%20GENERAL.csv"
dataframe = pd.read_csv(uri)

# remover atributos desnecessários
dataframe.drop(columns=['CUST_ID', 'TENURE'], inplace=True)

# achar dados que estão faltando
missing = dataframe.isna().sum
# print(missing)

# substituir os valore faltantes pela mediana
dataframe.fillna(dataframe.median(), inplace=True)

# normalizar os dados
values = Normalizer().fit_transform(dataframe.values)
# print(values)

# criando os clusters
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(values)

# achar o coeficiente de silhueta
labels = kmeans.labels_
silhouette = silhouette_score(values, labels=labels, metric="euclidean")
print("Coeficiente de silhueta: ", silhouette)

# achar o índice de Davies-Bouldin
dbs = davies_bouldin_score(values, labels)
print("Coeficiente de Davies-Bouldin: ", dbs)

# achar o índice de Calinski-Harabasz
calinski = calinski_harabasz_score(values, labels=labels)
print("Coeficiente de Calinski-Harabasz: ", calinski)


def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(dataset)
    s = silhouette_score(dataset, labels=labels, metric="euclidean")
    dbs = davies_bouldin_score(dataset, labels)
    calinski = calinski_harabasz_score(dataset, labels=labels)
    return s, dbs, calinski


# visualizando 2 atributos do cluster
plt.scatter(dataframe['PURCHASES'], dataframe['PAYMENTS'], c=labels, s=5, cmap='rainbow')
plt.xlabel("Valor total pago")
plt.ylabel("valor total gasto")
plt.show()

dataframe["cluster"] = labels
dataframe.groupby("cluster")

centroids = kmeans.cluster_centers_
max = len(centroids[0])

for i in range(max):
    print(dataframe.columns.values[i], "{:.4f}".format(centroids[:, i].var()))

#para reduzir o número de atributos, selecionar apenas os que
#tem maior covariância
