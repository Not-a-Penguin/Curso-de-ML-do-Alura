import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

uri = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"

filmes = pd.read_csv(uri)

filmes.columns = ['filme_id', 'titulo', 'generos']

scaler = StandardScaler()

generos = filmes.generos.str.get_dummies()
dados_dos_filmes = pd.concat([filmes, generos], axis=1)

generos_escalados = scaler.fit_transform(generos)

modelo = KMeans(n_clusters=3)
modelo.fit(generos_escalados)
print(modelo.labels_)