import pandas as pd
import numpy as np  
from collections import Counter 

arquivo = 'buscas.csv'
df = pd.read_csv(arquivo)
#É importante anotar as variáveis que está usando nos testes e sua acurácia.
X_df = df[['home', 'busca', 'logado']]
#print(X_df.head())
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
#print(Xdummies.head())
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1
porcentagem_de_validacao = 0.1  

tamanho_de_treino = int(porcentagem_de_treino*len(Y))
tamanho_de_teste = int(porcentagem_de_teste*len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste
fim_de_teste = tamanho_de_treino + tamanho_de_teste

treino_x = X[0:tamanho_de_treino]
treino_y = Y[0:tamanho_de_treino]

teste_x = X[tamanho_de_treino:fim_de_teste]
teste_y = Y[tamanho_de_treino:fim_de_teste]

validacao_x = X[fim_de_teste:]
validacao_y = Y[fim_de_teste:]

def fit_and_predict(nome, modelo, treino_x, treino_y, teste_x, teste_y):
    modelo.fit(treino_x, treino_y)
    previsoes = modelo.predict(teste_x)
    acertos = previsoes == teste_y #compara os valores iguais nas previsoes e no teste_y
    total_de_acertos = sum(acertos)#no teste  ##pega a quantidade de acertos já que sum() soma os valores True
    total_de_elementos = len(teste_x) #no teste
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto) 
    ##mostrando a acurácia
    print(msg)
    return taxa_de_acerto

def teste_real(modelo, validacao_x, validacao_y):
    previsoes = modelo.predict(validacao_x)
    acertos = previsoes == validacao_y #compara os valores iguais nas previsoes e no teste_y
    total_de_acertos = sum(acertos)#no teste  ##pega a quantidade de acertos já que sum() soma os valores True
    total_de_elementos = len(validacao_x) #no teste
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "taxa de acerto do algoritmo vencedor no mundo real: {0}".format(taxa_de_acerto) 
    ##mostrando a acurácia
    print(msg)

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_x, treino_y, teste_x, teste_y)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoostClassifier = AdaBoostClassifier()   
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoostClassifier, treino_x, treino_y, teste_x, teste_y)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoostClassifier

teste_real(vencedor, validacao_x, validacao_y)

#vendo eficácia do algoritmo base
# o método values retorna as quantidades de 'sim' e 'não'
acerto_base = max(Counter(teste_y).values()) # pega o máximo valor do mapa de valores criado pelo método Counter de Collections 
taxa_de_acerto_base = 100.0 * acerto_base/len(teste_y)
print("Taxa de acerto base: %.1f%%" % taxa_de_acerto_base)
total_de_elementos = len(teste_x) #no teste
print("Total de elementos: %d " % total_de_elementos)
