import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

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
