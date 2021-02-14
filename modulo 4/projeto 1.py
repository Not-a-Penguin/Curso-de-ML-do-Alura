import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

uri = "https://raw.githubusercontent.com/alura-cursos/reducao-dimensionalidade/master/data-set/exames.csv"
SEED = 123143
np.random.seed(SEED)
padronizador = StandardScaler()

resultados_exames = pd.read_csv(uri)

valores_exames = resultados_exames.drop(columns=['id', 'diagnostico'])
diagnostico = resultados_exames.diagnostico
valores_exames_v1 = valores_exames.drop(columns=['exame_33'])

padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data=valores_exames_v2, columns=valores_exames_v1.keys())


# classificador_bobo = DummyClassifier(strategy="most_frequent")
# classificador_bobo.fit(treino_x, treino_y)
# print("resultado da classificação do dummy classifier %.2f%%" % (classificador_bobo.score(teste_x, teste_y) * 100))


def grafico_violino(valores, inicio, fim):
    dados_plot = pd.concat([diagnostico, valores.iloc[:, inicio:fim]], axis=1)
    dados_plot = pd.melt(dados_plot, id_vars='diagnostico', var_name='exames', value_name='valores')

    plt.figure(figsize=(10, 10))
    plt.xticks(rotation=90)
    sns.violinplot(x='exames', y='valores', data=dados_plot, hue='diagnostico', split=True)


grafico_violino(valores_exames_v2, 10, 21)

valores_exames_v3 = valores_exames_v2.drop(columns=['exame_29', 'exame_4'])


def classificar(valores):
    SEED = 1234
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size=0.3)

    classificador = RandomForestClassifier(n_estimators=100)
    classificador.fit(treino_x, treino_y)

    print("resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y) * 100))


classificar(valores_exames_v3)
