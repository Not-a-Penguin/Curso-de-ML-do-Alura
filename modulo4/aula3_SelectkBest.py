import pandas as pd
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest #algoritmo para selecionar automaticamente
from sklearn.feature_selection import chi2 #função utilizada com selectkBest
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
'''
matriz_correlacao = valores_exames_v3.corr()
plt.figure(figsize=(17,15))
sns.heatmap(matriz_correlacao, annot=True, fmt=".1f")   
matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao>0.99]
matriz_correlacao_v2 = matriz_correlacao_v1.sum()
variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2>1]
#print(variaveis_correlacionadas)
#plt.show()'''
#classificar(valores_exames_v2)
#valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())
#print(valores_exames_v4.head())
#classificar(valores_exames_v4)
#valores_exames_v5 = valores_exames_v3.drop(columns = ["exame_3", "exame_24"])
#classificar(valores_exames_v5)

valores_exames_v6 = valores_exames_v1.drop(columns=['exame_3', 'exame_4', 'exame_29', 'exame_24']) #retira colunas que não afetam o score

def classificar(valores):
    SEED = 20
    random.seed(SEED)
    selecionar_kmelhores = SelectKBest(score_func=chi2, k=5) #cria o modelo e escolhe apenas 5 colunas como resultado
    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size = 0.3)
    selecionar_kmelhores.fit(treino_x, treino_y) #aplica os dados de treino para que os 5 melhores sejam escolhidos
    treino_kbest = selecionar_kmelhores.transform(treino_x) #transforma o treino
    teste_kbest = selecionar_kmelhores.transform(teste_x) #transforma o teste
    classificador = RandomForestClassifier(n_estimators=100) 
    classificador.fit(treino_kbest, treino_y) #aplica os dados no classificador
    resultado = classificador.score(teste_kbest, teste_y)*100 
    print("Resultado do classificador= %.2f%%" % resultado)


classificar(valores_exames_v6)