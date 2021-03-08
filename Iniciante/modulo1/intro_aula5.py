import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.dummy import DummyClassifier

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
modelo = SVC()

scaler = StandardScaler()  #instância a classe

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              #random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de treino e teste
                                                        )

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x),len(raw_teste_x)))

scaler.fit(raw_treino_x) #realiza o treino do processo de mudar a escala com dados do raw_treino_X
treino_x = scaler.transform(raw_treino_x) # muda a escala definitivamente para uma nova escala
teste_x = scaler.transform(raw_teste_x) # muda a escala definitivamente para uma nova escala

modelo.fit(treino_x, treino_y) #faz o modelo treinar com os dados de treino
previsoes = modelo.predict(teste_x) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia) 

##Gerar algoritmo de Baseline Automaticamente com o método 'stratified'
dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(treino_x, treino_y)
acuracia_dummy_stratified = dummy_stratified.score(treino_x, treino_y) * 100
print("A acurácia do baseline foi %.2f%%" % acuracia_dummy_stratified)

##Gerar algoritmo de Baseline Automaticamente com o método 'most_frequent'
dummy_mostfrequent = DummyClassifier(strategy='most_frequent')
dummy_mostfrequent.fit(treino_x, treino_y)
acuracia_mostfrequent = dummy_mostfrequent.score(treino_x, treino_y) * 100
print("A acurácia do baseline foi %.2f%%" % acuracia_mostfrequent)