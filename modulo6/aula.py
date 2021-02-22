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

s, dbs, calinski = algoritmo_cluster(3,valores_normalizados)
print(" Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s, dbs, calinski))

s_2, dbs_2, calinski_2 = algoritmo_cluster(5,valores_normalizados)
print(" Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s_2, dbs_2, calinski_2))

s_3, dbs_3, calinski_3 = algoritmo_cluster(10,valores_normalizados)
print(" Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s_3, dbs_3, calinski_3))

df.count() #Verificando a quantidade de dados que é 8950

import numpy as np

dados_aleatorios = np.random.rand(8950,16) #geramos uma matriz uniforme (ou seja, agrupada de um jeito padrão) do tamanho da nossa de dados, mas com valores aleatórios

s_r, dbs_r, calinski_r = algoritmo_cluster(5, dados_aleatorios) #aplicamos a matriz random no modelo para clusterização

#Abaixo podemos comparar nossa clusterização com a de dados aleatórios
print(" Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s_r, dbs_r, calinski_r))
print("\n Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s_2, dbs_2, calinski_2))

set1, set2, set3 = np.array_split(valores_normalizados, 3) #dividindo nossos dados em três
#aplicando cado grupo da linha acima para comparar nossos índices e compravar a estabilidade 
s1, dbs1, calinski1 = algoritmo_cluster(5, set1)
s2, dbs2, calinski2 = algoritmo_cluster(5, set2)
s3, dbs3, calinski3 = algoritmo_cluster(5, set3)

#Percebemos que para cada grupo, os índices não variam tanto, então nossa clusterização é estável
print(" Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s1, dbs1, calinski1))
print("\n Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s2, dbs2, calinski2))
print("\n Coeficiente de silhueta: {0}\n Índice de Davies-Boulding: {1}\n Índice de Calinski-Harabasz: {2}".format(s3, dbs3, calinski3))

import matplotlib.pyplot as plt

plt.scatter(x= df['COMPRAS'], y= df['PAGAMENTOS'], c=labels_kmeans_0, s=5, cmap='rainbow')
plt.xlabel("valor total de pago")
plt.ylabel("valor total de gasto")
plt.show()

import seaborn as sns

df["cluster"] = labels_kmeans_0

#sns.pairplot(df[:0], hue="cluster")
df.groupby(df["cluster"]).describe()
centroides = modelo_kmeans_0.cluster_centers_ 
#print(centroides)

max =  len(centroides[0]) #pegando o número total de elementos

for i in range(max):
  print(df.columns.values[i], "{:.4f}\n".format(centroides[:,i].var()))

descricao = df.groupby("cluster")["BALANCE", "COMPRAS", "CASH_ADVANCE", "CREDIT_LIMIT", "PAGAMENTOS"]
n_clientes = descricao.size()
descricao = descricao.mean()
descricao['n_clientes'] = n_clientes
print(descricao)

df.groupby("cluster")["PRC_FULL_PAYMENT"].describe() #avalia o atributo de porcentagem de pagamento
#percebe-se que cluster 0 possui a melhor média de pagamento