import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri) #lê tabelas do tipo CSV (palavra, vírgula, palavra)
#print(dados.head()) #seleciona apenas os 5º termos da tabela por padrão

mapa = {  #Mapeia os nomes originais da tabela e como vc quer chamá-los agora
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa) #executa a renomeação

print(dados)
x = dados[["principal", "como_funciona", "contato"]] #seleciona como "dados x" as 3 primeiras colunas 
y = dados["comprou"] #seleciona como "classe y" a última coluna

SEED = 20 #número para diminuir a aleatoriedade da separação 

treino_x, teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de treino e teste
                                                        )

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x),len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y) #faz o modelo treinar com os dados de treino
previsoes = modelo.predict(teste_x) #testa o modelo com os dados de teste_X
acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia = %.2f%%" % acuracia)  #mostra a acurácia do modelo