import pandas as pd

uri = "https://raw.githubusercontent.com/alura-cursos/alura-clustering-validation/base-de-dados/CC%20GENERAL.csv"

df = pd.read_csv(uri)
#df.head()

df = df.drop(columns=['CUST_ID', 'TENURE'], axis=1)
#ou podemos usar a linha abaixo
#df.drop(columns=["CUST_ID", "TENURE"], inplace=True) #o 'inplace = True informa que a alteração deve ser feita  no df'
#df.head()

renomear = {
            "PURCHASES":"COMPRAS", 
            "PAYMENTS":"PAGAMENTOS"
}

df.rename(columns=renomear,inplace=True)

missing = df.isna().sum() #soma todos os valores NULL do df
print(missing) #mostra quantos elementos são NULL em cada feature

df.fillna(df.median(), inplace=True) #prenche os lugares vazios do df com a mediana dos valores de df
missing = df.isna().sum() #soma todos os valores NULL do df
print(missing) #mostra quantos elementos são NULL em cada feature

from sklearn.preprocessing import Normalizer

valores_normalizados = Normalizer().fit_transform(df.values)
valores_normalizados

from sklearn.cluster import KMeans

modelo_kmeans_0 = KMeans(n_clusters = 5, n_init = 10, max_iter = 300)
grupos = modelo_kmeans_0.fit_predict(valores_normalizados) #grupos gerados pelo KMeans

from sklearn import metrics

labels_kmeans_0 = modelo_kmeans_0.labels_ #atribui os valores dos dados uma label específica

coeficiente_silhueta = metrics.silhouette_score(valores_normalizados, labels_kmeans_0, metric='euclidean') #coeficiente de silhueta para o cluster

print("Coeficiente de silhueta: {0}".format(coeficiente_silhueta))


## DAVIES-BOULDING
dbs = metrics.davies_bouldin_score(valores_normalizados, labels_kmeans_0)
print("Índice de Davies-Bouldin: {0}".format(dbs))

## CAMINSKI-HARABASZ
calinski = metrics.calinski_harabasz_score(valores_normalizados, labels_kmeans_0)
print("Índice calinski_harabasz: {0}".format(calinski))

def algoritmo_cluster(n_clusters, dados):
  modelo_kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
  grupos = modelo_kmeans.fit_predict(dados)
  labels_kmeans = modelo_kmeans.labels_

  s = metrics.silhouette_score(dados, labels_kmeans, metric='euclidean')
  dbs = metrics.davies_bouldin_score(dados, labels_kmeans)
  calinski = metrics.calinski_harabasz_score(dados, labels_kmeans)

  return s, dbs, calinski