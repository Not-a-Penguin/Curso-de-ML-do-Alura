from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


## Minha abordagem inicial foi 90% treino e 10% teste
#obtive uma acurácia de 88.89%
X,Y = carregar_acessos()

treino_x = X[:90]
treino_y = Y[:90]

teste_x = X[-9:]
teste_y = Y[-9:]

#criando modelo
modelo = MultinomialNB()
modelo.fit(treino_x, treino_y)

resultado = modelo.predict(teste_x)

##medindo a acurácia
diferencas = resultado - teste_y

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)#no teste
total_de_elementos = len(teste_x) #no teste
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

##mostrando a acurácia
print(taxa_de_acerto)
print(total_de_elementos) 
