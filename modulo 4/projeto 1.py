import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
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
    violino_grafico = plt.figure(1)
    sns.violinplot(x='exames', y='valores', data=dados_plot, hue='diagnostico', split=True)


valores_exames_v3 = valores_exames_v2.drop(columns=['exame_29', 'exame_4'])


def classificar(valores):
    SEED = 1234
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size=0.3)

    classificador = RandomForestClassifier(n_estimators=100)
    classificador.fit(treino_x, treino_y)

    print("resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y) * 100))


grafico_violino(valores_exames_v2, 10, 21)

classificar(valores_exames_v3)

matriz_correlacao = valores_exames_v3.corr()
grafico_heatmap = plt.figure(2, figsize=(17, 15))
sns.heatmap(matriz_correlacao, annot=True, fmt=".1f")

matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao > 0.99]
matriz_correlacao_v2 = matriz_correlacao_v1.sum()

variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2 > 1]

valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())
valores_exames_v5 = valores_exames_v3.drop(columns=["exame_3", "exame_24"])

classificar(valores_exames_v5)

selecionar_kmelhores = SelectKBest(chi2, k=5)

SEED = 1234
np.random.seed(SEED)

valores_exames_v6 = valores_exames_v1.drop(columns=["exame_4", "exame_29", "exame_3", "exame_24"])
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size=0.3)

selecionar_kmelhores.fit(treino_x, treino_y)
treino_kbest = selecionar_kmelhores.transform(treino_x)
teste_kbest = selecionar_kmelhores.transform(teste_x)

classificador = RandomForestClassifier(n_estimators=100, random_state=1234)
classificador.fit(treino_kbest, treino_y)
print("resultado da kbest %.2f%%" % (classificador.score(teste_kbest, teste_y) * 100))

plt.figure(3)
matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_kbest))
sns.heatmap(matriz_confusao, annot=True, fmt="d").set(xlabel="Predição", ylabel="Real")

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size=0.3)
classificador = RandomForestClassifier(n_estimators=100, random_state=1234)
classificador.fit(treino_x, treino_y)

plt.figure(4)

selecionar_rfe = RFE(estimator=classificador, n_features_to_select=5, step=1)
selecionar_rfe.fit(treino_x, treino_y)
treino_rfe = selecionar_rfe.transform(treino_x)
teste_rfe = selecionar_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)
matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfe))
sns.heatmap(matriz_confusao, annot=True, fmt="d").set(xlabel="Predição", ylabel="Real")
print("resultado da rfe %.2f%%" % (classificador.score(teste_rfe, teste_y) * 100))

plt.figure(5)

selecionar_rfecv = RFECV(estimator=classificador, cv=5, step=1, scoring="accuracy")
selecionar_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionar_rfecv.transform(treino_x)
teste_rfecv = selecionar_rfecv.transform(teste_x)
classificador.fit(treino_rfecv, treino_y)
matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfecv))
print("resultado da rfecv %.2f%%" % (classificador.score(teste_rfecv, teste_y) * 100))
sns.heatmap(matriz_confusao, annot=True, fmt="d").set(xlabel="Predição", ylabel="Real")

plt.show()
