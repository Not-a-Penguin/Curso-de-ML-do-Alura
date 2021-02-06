from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd

dados_x, _ = make_blobs(n_samples=1000, n_features=2, random_state=7) #gera alguns dados em duas dimensões para conseguirmos visualizar seus centroides.
dados_x = pd.DataFrame(dados_x, columns=['coluna1', 'coluna2']) #cria um dataframe com os 1000 de 2 dimensões 
modelo = DBSCAN()
grupos = modelo.fit_predict(dados_x)
plt.scatter(x=dados_x.coluna1, y=dados_x.coluna2, 
            c=grupos,
           cmap='viridis')
plt.show()