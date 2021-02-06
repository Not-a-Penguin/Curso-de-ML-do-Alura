from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

dados_x, _ = make_blobs(n_samples=1000, n_features=2, random_state=7) #gera alguns dados em duas dimensões para conseguirmos visualizar seus centroides.
dados_x = pd.DataFrame(dados_x, columns=['coluna1', 'coluna2']) #cria um dataframe com os 1000 de 2 dimensões 

plt.scatter(x=dados_x.coluna1, y=dados_x.coluna2) #plota os dados de forma dispersa
modelo = KMeans(n_clusters=3) #separa os dados em 3 grupos
grupos = modelo.fit_predict(dados_x) #faz a previsão com o modelo

plt.scatter(x=dados_x.coluna1, y=dados_x.coluna2, 
            c=grupos,
           cmap='viridis')
centroides = modelo.cluster_centers_ #acha as coordenadas do centroide do modelo

plt.scatter(dados_x.coluna1, dados_x.coluna2,
            c=grupos,
           cmap='viridis')

plt.scatter(centroides[:, 0], centroides[:, 1],
           marker='X', s=169, linewidths=5,
           color='g', zorder=8)
plt.show()