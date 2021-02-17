import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
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
valores_exames_v5 = valores_exames_v3.drop(columns = ["exame_3", "exame_24"])
valores_exames_v6 = valores_exames_v1.drop(columns=['exame_3', 'exame_4', 'exame_29', 'exame_24'])

SEED = 20
random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)
classificador = RandomForestClassifier(n_estimators=100, random_state=SEED)
classificador.fit(treino_x, treino_y)

selecionador_rfecv = RFECV(estimator = classificador, cv=5,step=1,scoring="accuracy")
selecionador_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionador_rfecv.transform(treino_x)
teste_rfecv = selecionador_rfecv.transform(teste_x)

classificador.fit(treino_rfecv, treino_y)

previsao = classificador.predict(teste_rfecv)
resultado = classificador.score(teste_rfecv, teste_y)*100
print("Resultado do classificador= %.2f%%" % resultado)


pca = PCA(n_components=2)

#valores_exames_v7 = selecionador_rfecv.transform(valores_exames_v6)

#valores_exames_v8 = pca.fit_transform(valores_exames_v6) #funciona mas exames_v6 não está padronizado
valores_exames_v8 = pca.fit_transform(valores_exames_v5)
plt.figure(figsize=(14,8))
sns.scatterplot(x = valores_exames_v8[:,0], y = valores_exames_v8[:,1], hue = diagnostico)
plt.show()

