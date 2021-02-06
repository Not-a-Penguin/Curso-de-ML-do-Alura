import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns       #importa a biblioteca
from sklearn.model_selection import train_test_split #importa o método para separação automática
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

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

SEED = 20 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED) #tira a necessidade de passar o SEED como paraâmetros de train_test_split() e LinearSVC()
modelo = LinearSVC()

treino_x, teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              #random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de treino e teste
                                                        )

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x),len(teste_x)))
modelo.fit(treino_x, treino_y) #faz o modelo treinar com os dados de treino
previsoes = modelo.predict(teste_x) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia)  

previsoes_de_baseline = np.ones(540) # gera uma matriz de 1 para aferir a qualidade do modelo
acuracia = accuracy_score(teste_y, previsoes_de_baseline)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do alogoritmo de baseline = %.2f%%" % acuracia)  
sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y, data=teste_x) #analisar as classificações

x_min = teste_x.horas_esperadas.min() #determina que quer o valore minimo de "horas_esperadas"
x_max = teste_x.horas_esperadas.max() #determina que quer o valore máximo de "horas_esperadas"
y_min = teste_x.preco.min() #determina que quer o valore minimo de "preco"
y_max = teste_x.preco.max() #determina que quer o valore máximo de "preco"

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max-x_min) / pixels) #determina o passo de eixo x com arange
eixo_y = np.arange(y_min, y_max, (y_max-y_min) / pixels) #determina o passo do eixo y com arange
xx, yy = np.meshgrid(eixo_x, eixo_y) #mescla os elementos do eixos
pontos = np.c_[xx.ravel(), yy.ravel()] #concatena os elementos do eixos para formar os pontos

Z = modelo.predict(pontos) #aplica o modelo nos pontos 
Z = Z.reshape(xx.shape) #formata o resultado do modelo para o formato de xx (100x100)

plt.contourf(xx, yy, Z, alpha=0.2) #cria uma área para vizualizar a curva  e alpha seta a transparência da área pintada.
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1) #plota de os testes de forma dispersa baseando a cor no teste_y (resultado/classificação)  
plt.show()