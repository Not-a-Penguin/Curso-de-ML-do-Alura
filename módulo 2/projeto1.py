import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns

uri = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"

filmes = pd.read_csv(uri)

filmes.columns = ['filme_id', 'titulo', 'generos']

scaler = StandardScaler()

generos = filmes.generos.str.get_dummies()
dados_dos_filmes = pd.concat([filmes, generos], axis=1)

generos_escalados = scaler.fit_transform(generos)

modelo = KMeans(n_clusters=3)
modelo.fit(generos_escalados)
grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
grupos.transpose().plot.bar(subplots=True, figsize=(25, 25), sharex=False)

grupo = 0
filtro = modelo.labels_ == grupo
dados_dos_filmes[filtro].sample(10)

tsne = TSNE()
visualizacao = tsne.fit_transform(generos_escalados)

sns.set(rc={'figure.figsize': (13, 13)})

a = sns.scatterplot(x=visualizacao[:, 0], y=visualizacao[:, 1], hue=modelo.labels_, palette=sns.color_palette('Set1', 3))
plt.show()