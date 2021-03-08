from sklearn.model_selection import train_test_split #importa o método para separação automática
from sklearn.tree import DecisionTreeClassifier # usa o estimador DecisionTreeClassifier de classificação e que permite análise das tomadas de decisão
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler #importa classe StandardScaler que faz uma nova escala igual para os nossos dados 
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz #módulo export_graphviz traz as informações das  tomadas de decisão
from datetime import datetime
import graphviz  #biblioteca para plotar o gráfico

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(uri)
renomear = {
        'mileage_per_year':'milhas_por_ano',
        'model_year':'ano_do_modelo',
        'price':'preco',
        'sold':'vendido'
}

dados = dados.rename(columns=renomear)

troca = {
        'yes':1,
         'no':0
}
dados.vendido= dados.vendido.map(troca)

ano_atual = datetime.today().year
dados['idade_do_carro'] = ano_atual - dados.ano_do_modelo

dados['km_por_ano'] = dados.milhas_por_ano * 1.609344
dados = dados.drop(columns = ["Unnamed: 0", "milhas_por_ano", "ano_do_modelo"], axis=1)

x = dados[["preco","idade_do_carro","km_por_ano"]]
y =  dados["vendido"]

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)
modelo = DecisionTreeClassifier(max_depth=3) #inicializa a árvore de decisão com o parâmetro de "profundidade" em 3, ou seja, três camadas

#scaler = StandardScaler()  #instância a classe

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              #random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de treino e teste
                                                        )

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x),len(raw_teste_x)))

#no algoritmo de àrvore de decisão não é preciso ajustar as escalas dos eixo, então as 3 linhas a seguir não são necessárias 
'''scaler.fit(raw_treino_x) #realiza o treino do processo de mudar a escala com dados do raw_treino_X
treino_x = scaler.transform(raw_treino_x) # muda a escala definitivamente para uma nova escala
teste_x = scaler.transform(raw_teste_x) # muda a escala definitivamente para uma nova escala
'''
modelo.fit(raw_treino_x, treino_y) #faz o modelo treinar com os dados cru de treino
previsoes = modelo.predict(raw_teste_x) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia) 

features = x.columns #nomea o nome das features como as colunas de x, ou seja, "preco", "km_por_ano", "idade_do_carro"
dot_data = export_graphviz(modelo, #recebe as informaçãoes da tomada de decisão para o modelo
                          out_file=None, #padrão
                          feature_names=features, #seta o nome das features que devem aparecer no gráfico aos invés de X[0], X[1], ...
                          filled = True, #Ativa o modo colorido
                          rounded = True, #Ativa o arredondamento das caixas
                           class_names=['não', 'sim']) #renomeia os resultados das classes, ou seja, "vendido" "0" ou "1" vira 
#                                                                                                    "vendido" "não" ou "sim" 
grafico = graphviz.Source(dot_data, format='png') #plota o gráfico de acordo com a informação do dot_data
grafico.view()
#informações na Tree
#Samples: quantidade de dados sendo analisados para a tomada de decisão
#Gini: método para tomada de decisão 