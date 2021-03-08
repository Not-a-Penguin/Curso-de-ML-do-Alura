import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

uri = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"
dados_filmes = pd.read_csv(uri)
renomear = {
            'movieId':'id_filme',
            'title':'título',
            'genres':'generos'
}
dados_filmes = dados_filmes.rename(columns=renomear)
generos_filmes = dados_filmes.generos.str.get_dummies() #transfomra a coluna "gêneros" em várias variáveis categóricas para facilitar o tratamento dos dados
#generos_filmes.head()
dados = pd.concat([dados_filmes, generos_filmes], axis=1)  #usa o método de concatenar do pandas para juntar as novas colunas formadas por var. categóricas às colunas já existentes
#dados = dados.drop(columns="generos", axis=1)

#Cria o modelo e treina com hierarquia
scaler = StandardScaler()  #instancia o escalador
generos_filmes_escalados = scaler.fit_transform(generos_filmes) #reescala os dados da coluna "gêneros" para o tratamento dos dados com o estimador

modelo = AgglomerativeClustering(n_clusters = 17)
grupos = modelo.fit_predict(generos_filmes_escalados)
tsne = TSNE()
visualizacao = tsne.fit_transform(generos_filmes_escalados)

sns.set(rc={'figure.figsize':(13,13)})

sns.scatterplot(x=visualizacao[:,0],
                y=visualizacao[:,1],
                hue= grupos)
                #palette=sns.color_palette('Set1',17))

plt.show()