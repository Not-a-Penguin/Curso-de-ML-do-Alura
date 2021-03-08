import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
import seaborn as sns
import matplotlib.pyplot as plt 

uri = "https://raw.githubusercontent.com/alura-cursos/reducao-dimensionalidade/master/data-set/exames.csv"

resultados_exames = pd.read_csv(uri)

valores_exames = resultados_exames.drop(columns = ['id', 'diagnostico'])
diagnostico = resultados_exames.diagnostico
valores_exames_v1 = valores_exames.drop(columns='exame_33')
padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data = valores_exames_v2,
                                columns = valores_exames_v1.keys())
valores_exames_v3 = valores_exames_v2.drop(columns=["exame_4", "exame_29"])
valores_exames_v6 = valores_exames_v1.drop(columns=['exame_3', 'exame_4', 'exame_29', 'exame_24'])

SEED = 20
random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)
classificador = RandomForestClassifier(n_estimators=100, random_state=SEED)
classificador.fit(treino_x, treino_y)

selecionador_rfecv = RFECV(estimator = classificador, cv=5,step=1,scoring="accuracy") #cria o modelo para selecionar o número de features que gera o melhor resultado
selecionador_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionador_rfecv.transform(treino_x)
teste_rfecv = selecionador_rfecv.transform(teste_x)
print(treino_rfecv)

classificador.fit(treino_rfecv, treino_y)

previsao = classificador.predict(teste_rfecv)
resultado = classificador.score(teste_rfecv, teste_y)*100
print("Resultado do classificador= %.2f%%" % resultado)

'''
matriz_confusao = confusion_matrix(teste_y, previsao)
print(matriz_confusao)
plt.figure(figsize=(10,8))
sns.set(font_scale=2)
sns.heatmap(matriz_confusao, annot=True, fmt="d").set(xlabel= "Predição", ylabel="Real")
plt.show()
'''

#print(selecionador_rfecv.n_features_) #mostra quantas features foram selecionadas
#print(treino_x.columns[selecionador_rfecv.support_]) #mostra um lista com o nomes das features escolhidas

x_limit = len(selecionador_rfecv.grid_scores_) #determina o número de features
y_values = selecionador_rfecv.grid_scores_  #retorna as acurácias de cada feature

plt.figure(figsize=(10,8))
plt.xlabel = "N° de exames"
plt.ylabel = "Acurácia"
plt.plot(range(1,x_limit+1),y_values)
plt.grid()
plt.show()