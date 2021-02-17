import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
print("coeficiente de silhueta: ", silhouette)

#achar o índice de Davies-Bouldin
dbs = davies_bouldin_score(values, labels)
print("Coeficiente de Davies-Bouldin: ", dbs)


