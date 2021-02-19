import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

uri = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"

filmes = pd.read_csv(uri)

filmes.columns = ['filme_id', 'titulo', 'generos']

scaler = StandardScaler()

generos = filmes.generos.str.get_dummies()
dados_dos_filmes = pd.concat([filmes, generos], axis=1)

generos_escalados = scaler.fit_transform(generos)


# modelo = KMeans(n_clusters=3)
# modelo.fit(generos_escalados)
# grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
# grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False)
#
# grupo = 0
# filtro = modelo.labels_ == grupo
# dados_dos_filmes[filtro].sample(10)
#
# tsne = TSNE()
# visualizacao = tsne.fit_transform(generos_escalados)
#
# sns.set(rc={'figure.figsize': (13, 13)})
#
# a = sns.scatterplot(x=visualizacao[:, 0], y=visualizacao[:, 1], hue=modelo.labels_, palette=sns.color_palette('Set1', 3))

# parte 2

# modelo = KMeans(n_clusters=20)
# modelo.fit(generos_escalados)
# grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
# grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False, rot=0)

def kmeans(numero_de_clusters, generos):
    modelo = KMeans(n_clusters=numero_de_clusters)
    modelo.fit(generos)
    return [numero_de_clusters, modelo.inertia_]


# kmeans(20, generos_escalados)

# resultado = [kmeans(numero_de_grupos, generos_escalados) for numero_de_grupos in range(1, 41)]
# print(resultado)
# resultados = pd.DataFrame(resultado, columns=['grupos', 'inertia'])
#
# resultados.inertia.plot()
# plt.show()

# modelo = KMeans(n_clusters=17)
# modelo.fit(generos_escalados)
#
# grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
# grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False, rot=0)
#
# modelo = AgglomerativeClustering(n_clusters=17)
# grupos = modelo.fit_predict(generos_escalados)
# grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
# grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False, rot=0)
# tsne = TSNE()
# visualizacao = tsne.fit_transform(generos_escalados)
# sns.scatterplot(x=visualizacao[:,0], y=visualizacao[:, 1])
# plt.show()

modelo = KMeans(n_clusters=17)
modelo.fit(generos_escalados)
grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False, rot=0)

matriz_de_distancia = linkage(grupos)
dendrograma = dendrogram(matriz_de_distancia)
