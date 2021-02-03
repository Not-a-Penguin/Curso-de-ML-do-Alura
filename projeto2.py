# ler um arquivo csv

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()

dados = pd.read_csv(
    "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv")

mapa = {
    'home': 'principal',
    'how_it_works': "como_funciona",
    'contact': 'contato',
    'bought': 'comprou'
}

dados = dados.rename(columns=mapa)

x = dados[['principal', 'como_funciona', 'contato']]
y = dados['comprou']

treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)

print(acuracia)