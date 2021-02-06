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

modelo = KMeans(n_clusters=3) #cria um modelo com três grupos
modelo.fit(generos_filmes_escalados) #treina o modelo com os filmes depois do escalonamento
print(f'grupos {modelo.labels_}')  #mostra que os grupos já estão separados conforme 3 grupos, onde os valores vão de 1 a 3, por exemplo: "infatil", "terro", "ação"
print(generos_filmes.columns) #mostra a colunas usados no modelo
print(modelo.cluster_centers_) #mostra "pesos" de cada variável categórica nos filmes
raw_grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos_filmes.columns) #cria um dataframe grupos x V. categ. com os "pesos"
grupos = raw_grupos.transpose() #Cria uma matriz transposta de "raw_grupos" para facilitar a vizualização
#plota o dataframe transposto
gráfico = grupos.plot.bar(subplots=True, #separa o plot dos grupos
                figsize=(25,25), #configura o tamanho da figura
                sharex=False) #não compartilha a informação do eixo x para facilitar a visualização

grupo = 1 #variável para testar o modelo
filtro = modelo.labels_== grupo # usa a variável de teste para comparar com todos os rótulos dos filmes
dados[filtro].sample(10) #aplica o filtro nos dados do filme para 10 amostras
#mostra os filmes já agrupados por gêneros semelhantes

tsne = TSNE()
visualizacao = tsne.fit_transform(generos_filmes_escalados)

sns.set(rc={'figure.figsize':(13,13)})

sns.scatterplot(x=visualizacao[:,0],
                y=visualizacao[:,1],
                hue= modelo.labels_,
                palette=sns.color_palette('Set1',3))
plt.show()