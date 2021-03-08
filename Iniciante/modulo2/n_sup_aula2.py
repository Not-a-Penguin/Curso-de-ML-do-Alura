import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

scaler = StandardScaler()  #instancia o escalador
generos_filmes_escalados = scaler.fit_transform(generos_filmes) #reescala os dados da coluna "gêneros" para o tratamento dos dados com o estimador

modelo = KMeans(n_clusters=20)
modelo.fit(generos_filmes_escalados)
raw_grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos_filmes.columns) 
grupos = raw_grupos.transpose() #Cria uma matriz transposta de "raw_grupos" para facilitar a vizualização
#plota o dataframe transposto
gráfico = grupos.plot.bar(subplots=True, #separa o plot dos grupos
                figsize=(25,50), #configura o tamanho da figura
                sharex=False, #não compartilha a informação do eixo x para facilitar a visualização
                rot=0) #deixa a legendas na horizontal

grupo = 1 #variável para testar o modelo
filtro = modelo.labels_== grupo # usa a variável de teste para comparar com todos os rótulos dos filmes
dados[filtro].sample(5) #aplica o filtro nos dados do filme para 10 amostras
#mostra os filmes já agrupados por gêneros semelhantes
##### ELBOW METHOD
def kmeans(numero_de_clusters, generos_filmes):
    modelo = KMeans(n_clusters=numero_de_clusters)
    modelo.fit(generos_filmes)
    erro = modelo.inertia_
    return [numero_de_clusters, erro]

resultado = [kmeans(numero_de_grupos, generos_filmes_escalados) for numero_de_grupos in range(1,41)]
resultado = pd.DataFrame(resultado,
             columns=['grupos','erro'])

resultado.erro.plot(xticks = resultado.grupos)
#####

plt.show()