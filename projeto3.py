import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv")

modelo = LinearSVC()
SEED = 20

a_renomear = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizado'
}

dados = dados.rename(columns=a_renomear)

troca = {
    0: 1,
    1: 0
}

dados['finalizado'] = dados.nao_finalizado.map(troca)

sns.scatterplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado")

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)

previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100
print("Acurácia do algoritmo de baseline: ", acuracia)

