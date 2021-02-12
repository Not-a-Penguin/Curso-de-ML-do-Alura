##from dados import carregar_buscas
##X, Y =  carregar_buscas()
#print(X[0])

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import numpy as np  

arquivo = 'buscas.csv'
df = pd.read_csv(arquivo)
X_df = df[['home', 'busca', 'logado']]
#print(X_df.head())
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
#print(Xdummies.head())
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino*len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_x = X[:tamanho_de_treino]
treino_y = Y[:tamanho_de_treino]

teste_x = X[-tamanho_de_teste:]
teste_y = Y[-tamanho_de_teste:]

#vendo eficácia do algoritmo base
'''
acerto_de_um =  sum(Y);
acerto_de_zero = len(Y) - acerto_de_um
'''
acerto_de_um = len(Y[Y==1])
acerto_de_zero = len(Y[Y==0])
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero)/len(Y)
print("Taxa de acerto base: %.1f" % taxa_de_acerto_base)
 
modelo = MultinomialNB()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

diferencas = previsoes - teste_y

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)#no teste
total_de_elementos = len(teste_x) #no teste
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

##mostrando a acurácia
print("taxa de acerto: %.1f" % taxa_de_acerto)



