from sklearn.model_selection import train_test_split #importa o método para separação automática
from sklearn.svm import SVC # usa o estimador SVC de classificação
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #importa classe StandardScaler que faz uma nova escala igual para os nossos dados 
import numpy as np
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados  =pd.read_csv(uri)
renomear = {
          "unfinished":"Incompleto",
          "expected_hours":"horas_esperadas",
          "price":"preco"  
}

dados = dados.rename(columns=renomear)
troca = {         # faz a troca de 0 por 1 na nova coluna
        0:1,
        1:0
}

dados["completo"] = dados.Incompleto.map(troca) #cria uma nova coluna com a troca feita
x = dados[['horas_esperadas', 'preco']]
y = dados['completo']

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)
modelo = SVC()

scaler = StandardScaler()  #instância a classe

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              #random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de treino e teste
                                                        )

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x),len(raw_teste_x)))

scaler.fit(raw_treino_x) #realiza o treino do processo de mudar a escala com dados do raw_treino_X
treino_x = scaler.transform(raw_treino_x) # muda a escala definitivamente para uma nova escala
teste_x = scaler.transform(raw_teste_x) # muda a escala definitivamente para uma nova escala

modelo.fit(treino_x, treino_y) #faz o modelo treinar com os dados de treino
previsoes = modelo.predict(teste_x) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia)  
#agora que treino_x e teste_x foram reescalonados, o nome das tabelas de raw_treino_x e raw_teste_w mudou, pq agora não
# é mais um Pandas DataFrame e sim um array de arrays

data_x = teste_x[:,0] #usa-se esse métdodo para escolher a 1º coluna (antiga "horas_esperadas") em todas as linhas 
data_y = teste_x[:,1] #usa-se esse métdodo para escolher a 2º coluna (antiga "precos") em todas as linhas

x_min = data_x.min() #determina que quer o valore minimo da 1º coluna
x_max = data_x.max() #determina que quer o valore máximo da 1º coluna
y_min = data_y.min() #determina que quer o valore minimo da 2º coluna
y_max = data_y.max() #determina que quer o valore máximo da 2º coluna

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max-x_min) / pixels) #determina o passo de eixo x com arange
eixo_y = np.arange(y_min, y_max, (y_max-y_min) / pixels) #determina o passo do eixo y com arange

xx, yy = np.meshgrid(eixo_x, eixo_y) #mescla os elementos do eixos
pontos = np.c_[xx.ravel(), yy.ravel()] #concatena os elementos do eixos para formar os pontos
Z = modelo.predict(pontos) #aplica o modelo nos pontos 
Z = Z.reshape(xx.shape) #formata o resultado do modelo para o formato de xx (100x100)

plt.contourf(xx, yy, Z, alpha=0.2) #cria uma área para vizualizar a curva  e alpha seta a transparência da área pintada.
plt.scatter(data_x, data_y, c=teste_y, s=1) #plota os testes de forma dispersa baseando a cor no teste_y (resultado/classificação)  
plt.show()
