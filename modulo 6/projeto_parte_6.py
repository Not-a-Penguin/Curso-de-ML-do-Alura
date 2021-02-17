import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#importando os dados e criando o dataframe
uri = "https://raw.githubusercontent.com/alura-cursos/alura-clustering-validation/base-de-dados/CC%20GENERAL.csv"
dataframe = pd.read_csv(uri)

#remover atributos desnecessários
dataframe.drop(columns=['CUST_ID', 'TENURE'], inplace=True)

#achar dados que estão faltando
missing = dataframe.isna().sum
#print(missing)

#substituir os valore faltantes pela mediana
dataframe.fillna(dataframe.median(), inplace=True)

#normalizar os dados
values = Normalizer().fit_transform(dataframe.values)
#print(values)

#criando os clusters
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(values)

#achar o coeficiente de silhueta
labels = kmeans.labels_
silhouette = silhouette_score(values, labels=labels, metric="euclidean")
print("Coeficiente de silhueta: ", silhouette)

#achar o índice de Davies-Bouldin
dbs = davies_bouldin_score(values, labels)
print("Coeficiente de Davies-Bouldin: ", dbs)

#achar o índice de Calinski-Harabasz
calinski = calinski_harabasz_score(values, labels=labels)
print("Coeficiente de Calinski-Harabasz: ", calinski)

def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(dataset)
    s = silhouette_score(dataset, labels=labels, metric="euclidean")
    dbs = davies_bouldin_score(dataset, labels)
    calinski = calinski_harabasz_score(dataset, labels=labels)
    return s, dbs, calinski


s2, dbs2, calinski2 = clustering_algorithm(5, values)
print(s2, dbs2, calinski2)

#Silhouette é o índice mais popular

random_data = np.random.rand(8950, 16)
s, dbs, calinski = clustering_algorithm(5, random_data)

#dividir o cluster em 3 partes e calcular os índicies para
#cada um deles
set1, set2, set3 = np.array_split(values, 3)
s1, dbs1, calinski1 = clustering_algorithm(5, set1)
s2, dbs2, calinski2 = clustering_algorithm(5, set2)
s3, dbs3, calinski3 = clustering_algorithm(5, set3)
print(s1, dbs1, calinski1)
print(s2, dbs2, calinski2)
print(s3, dbs3, calinski3)

#como os valores são similares, indica que o cluster é
#estável
