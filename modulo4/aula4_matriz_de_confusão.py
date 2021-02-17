import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE #selecionador automático
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

selecionador_rfe = RFE(estimator = classificador, n_features_to_select=5,step=1) #determina para o selecionador que queer 5 features resultantes
selecionador_rfe.fit(treino_x, treino_y)
treino_rfe = selecionador_rfe.transform(treino_x)
teste_rfe = selecionador_rfe.transform(teste_x)

classificador.fit(treino_rfe, treino_y) #treina o modelo com os dados já selecionados

previsao = classificador.predict(teste_rfe)
resultado = classificador.score(teste_rfe, teste_y)*100
print("Resultado do classificador= %.2f%%" % resultado)

matriz_confusao = confusion_matrix(teste_y, previsao)   #cria uma matriz com as previsões feitas pelo modelo nas colunas e o resultado real nas linhas 
print(matriz_confusao)
plt.figure(figsize=(10,8))
sns.set(font_scale=2)
sns.heatmap(matriz_confusao, annot=True, fmt="d").set(xlabel= "Predição", ylabel="Real") #plota um mapa de calor para ilustrar a matriz de confusão
plt.show()