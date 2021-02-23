import plotly.graph_objects as go #biblioteca para visualizar a clusterização em 3D
import pandas as pd
import numpy as np
#from biokit.viz import corrplot  #biblioteca para plotar o gráfico da  matriz de correlação
import matplotlib.pyplot as plt

uri = "https://raw.githubusercontent.com/Not-a-Penguin/Curso-de-ML-do-Alura/Jo%C3%A3o/Wine.csv"
df = pd.read_csv(uri)

print("Os dados possuem ", df.shape[0], "amostras e", df.shape[1], "atributos")

labels_df = df["Customer_Segment"]
df = df.drop(columns="Customer_Segment")
renomear = {
        'Alcohol':'alcool',
        'Ash':'po',
        'Ash_Alcanity':'acalinidade_po',
         'Magnesium':'magnesio',
         'Color_Intensity':'intensidade_de_cor'
}

df = df.rename(columns=renomear)
df.describe()
matriz_correlacao = df.corr()
#matriz_correlacao
grafico_correlacao = corrplot.Corrplot(matriz_correlacao) #cria o gráfico da matriz de correlação
grafico_correlacao.plot(upper='ellipse', fontsize='x-large') #setamos o gráfico como crecente e com letras grandes
fig = plt.gcf()
fig.set_size_inches(13,8) #seta o tamanho da figura em polegadas
fig.show() 

"""
atributos  = df.columns
for atributo in atributos:
  df[atributo] = (df[atributo]-min(df[atributo]))/(max(df[atributo])-min(df[atributo]))

df.head()
"""

from sklearn import preprocessing #importando o normalizador
min_max_scaler = preprocessing.MinMaxScaler() #usando o método mínimo e máximo (0 a 1)
np_df = min_max_scaler.fit_transform(df) #aplicando os dados no normalizador e recebendo um array numpy
df = pd.DataFrame(np_df, columns= df.columns) #convertendo o array numpy para DataFrame

np_ndf = min_max_scaler.inverse_transform(df)
df_n_normalizado = pd.DataFrame(np_df, columns= df.columns)

from sklearn.cluster import KMeans

modelo = KMeans(n_clusters = 4)
modelo.fit(df)

labels = modelo.labels_
print(labels)

fig = go.Figure()
fig.add_trace(go.Scatter(x = df['intensidade_de_cor'], 
                         y = df['alcool'], 
                         mode= 'markers', 
                         marker= dict(color = modelo.labels_.astype(np.float)), 
                         text = labels))

fig.show()

modelo2 = KMeans(n_clusters=3)
modelo2.fit(df)

labels2 = modelo2.labels_
print(labels2)

fig_v2 = go.Figure()
fig_v2.add_trace(go.Scatter(x = df['intensidade_de_cor'], 
                         y = df['alcool'], 
                         mode= 'markers', 
                         marker= dict(color = modelo2.labels_.astype(np.float)), 
                         text = labels2))
fig_v2.show()

fig_v3 = go.Figure()
fig_v3.add_trace(go.Scatter3d(x = df['intensidade_de_cor'], #seta os valores recebidos em cada eixo com base nos atributos
                              y = df['alcool'],
                              z = df['Proline'],
                              mode = 'markers', #escolha o marcador como bolinha
                              marker = dict(color = modelo2.labels_.astype(np.float)), #indica a separação por cores baseado nos grupos gerados no KMEANS 
                              text  = labels2)) #seta o texto com base nas labels do KMEANS

fig_v3.update_layout(scene = dict( #insere o nome dos eixos no gráfico
    xaxis_title = 'Intensidade de Cor',
    yaxis_title = 'Álcool',
    zaxis_title = 'Proline'))

fig_v3.show()

centros = pd.DataFrame(modelo2.cluster_centers_)
centros.columns = df.columns
centros.head()

fig_v3.add_trace(go.Scatter3d(x = centros['intensidade_de_cor'],
                              y = centros['alcool'],
                              z = centros['Proline'],
                              mode = 'markers',
                              marker = dict(color = 'green'),
                              text = [0, 1, 2]))

fig_v3.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

faixa_n_clusters = [i for i in range (2,10)]
print(faixa_n_clusters)

valores_silhueta = []
for k in faixa_n_clusters:
  modelo = KMeans(n_clusters= k)
  labels = modelo.fit_predict(df)
  media_silhueta = silhouette_score(df, labels)
  valores_silhueta.append(media_silhueta)

fig = go.Figure()
fig.add_trace(go.Scatter(x = faixa_n_clusters, y = valores_silhueta))
fig.update_layout(
    title = "Valores de silhueta médios",
    xaxis_title = "Número de clusteres",
    yaxis_title = "Valor médio de silhueta",
)