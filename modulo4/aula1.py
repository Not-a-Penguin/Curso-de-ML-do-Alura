import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt 

uri = "https://raw.githubusercontent.com/alura-cursos/reducao-dimensionalidade/master/data-set/exames.csv"

resultados_exames = pd.read_csv(uri)

#print(resultados_exames.head())

SEED = 20
random.seed(SEED)
valores_exames = resultados_exames.drop(columns = ['id', 'diagnostico'])
diagnostico = resultados_exames.diagnostico
valores_exames_v1 = valores_exames.drop(columns='exame_33')

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data = valores_exames_v2,
                                columns = valores_exames_v1.keys())
#print(valores_exames_v2.head())
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v2, diagnostico, test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100)
classificador.fit(treino_x, treino_y)
resultado = classificador.score(teste_x, teste_y)*100
print("Resultado do classificador= %.2f%%" % resultado)

classificador_bobo = DummyClassifier(strategy="most_frequent")
classificador_bobo.fit(treino_x, treino_y)
resultado_bobo = classificador_bobo.score(teste_x, teste_y)*100
print("Resultado do classificador bobo= %.2f%%" % resultado_bobo)

dados_plot = pd.concat([diagnostico, valores_exames_v2.iloc[:,0:10]], axis = 1)
dados_plot = pd.melt(dados_plot, id_vars="diagnostico", 
                 var_name="exames",
                 value_name="valores")
#print(dados_plot.head())

plt.figure(figsize=(10,10))

sns.violinplot(x="exames", y="valores", hue = "diagnostico", data=dados_plot, split=True)
plt.xticks(rotation=90)
plt.show()