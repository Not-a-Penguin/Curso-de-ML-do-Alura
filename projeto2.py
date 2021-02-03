# ler um arquivo csv

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
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

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size= 0.25)
print(treino_x)

modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)