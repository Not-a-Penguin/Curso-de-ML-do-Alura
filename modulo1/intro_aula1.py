from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 
# features (1 - sim ou 0 - não )
#tem pelo longo?
#tem perna curta?
#faz auau?
#Dados
porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]

cao1 = [0,1,1]
cao2 = [1,0,1]
cao3 = [1,1,1]

#1 - é porco     0 - é cão 
treino_x = [porco1, porco2, porco3, cao1, cao2, cao3] #x - dados
treino_y = [1,1,1,0,0,0] #y - classes (labels / etiquetas)

modelo =  LinearSVC() #instancia a classe LinearSVC
modelo.fit(treino_x, treino_y) #manda o modelo aprender de forma supervisionada com os dados e as classes

animal_misterioso = [1,1,1]
modelo.predict([animal_misterioso]) #faz o modelo prever o resultado para testar sua aprendizagem
animal_misterioso1 = [1,1,1]
animal_misterioso2 = [1,1,0]
animal_misterioso3 = [0,1,1]

teste_x = [animal_misterioso1, animal_misterioso2, animal_misterioso3]
teste_y = [0,1,1] #resultado já pré-estabelecido

previsoes = modelo.predict(teste_x) #usa os resultados do modelo como previsões para testar acurácia
corretos = (previsoes == teste_y).sum() #soma o número de resultados corretos da igualdade das previões e dos pré-estabelecidos
total = len(teste_x)
taxa_de_acerto = (corretos/total)*100
print("Taxa de acerto: %.2f" % taxa_de_acerto)

##USANDO A BIBLIOTECA DO SKLEARN 
acuracia = accuracy_score(teste_y, previsoes) #passamos como param. os valores verdadeiros e depois as previsões do modelo
print("Taxa de acerto: %.2f" % (acuracia*100))
