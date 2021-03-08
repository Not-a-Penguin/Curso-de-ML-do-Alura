#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
dados = pd.read_csv(uri)
renomear = {
        'mileage_per_year':'milhas_por_ano',
        'model_year':'ano_do_modelo',
        'price':'preco',
        'sold':'vendido'
}

dados = dados.rename(columns=renomear)
#dados.head()

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
dados.head()


# # Gerando dados aleatórios
# É importante para treinar o modelo com grupos que podem surgir ao longo do tempo e garantir sua acurácia mesmo assim

# In[2]:


SEED = 5
np.random.seed(SEED)
dados["modelo_aleatorio"] = dados.idade_do_carro + np.random.randint(-2,3,size=10000) #+ minimo_valor
minimo_valor = abs(dados.modelo_aleatorio.min()) #para evitar q o menor valor seja negativo


dados.head()


# In[3]:


def imprime_resultados(resultados):
    media = resultado['test_score'].mean()
    dp = resultado['test_score'].std()
    print("Acurácia média: %.2f%%" % (media*100))
    print("Intervalo de Acurácia: %.2f - %.2f%%" % ((media-2*dp)*100, (media+2*dp)*100))


# In[4]:


dados_azar = dados.sort_values("vendido", ascending=True)
x_azar = dados_azar[["preco","idade_do_carro","km_por_ano"]]
y_azar =  dados_azar["vendido"]


# # O GroupKFold agrupa por grupos, então precisamos passar ao cross_validate a coluna 

# In[13]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = GroupKFold(n_splits=10)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x_azar, y_azar, cv=cv, groups=dados.modelo_aleatorio, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y) #classes para alocar os dados                                                             #random_state = SEED, #parâmetro para diminuir a aleatoriedade do métod                                                             test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total                                                             stratify = y # parâmetro para acertar a proporcionalidade da separação de                                                               )
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x),len(raw_teste_x)))


# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 5
np.random.seed(SEED)

escalador = StandardScaler()
escalador.fit(raw_treino_x)
treino_escalado = escalador.transform(raw_treino_x)
teste_escalado = escalador.transform(raw_teste_x)

modelo = SVC()

modelo.fit(treino_escalado, treino_y) #faz o modelo treinar com os dados cru de treino
previsoes = modelo.predict(teste_escalado) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia) 


# # Forma errada

# In[27]:


escalador.fit(x_azar)
x_azar_escalado = escalador.transform(x_azar)


# In[28]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = GroupKFold(n_splits=10)

modelo = SVC()
resultado = cross_validate(modelo,x_azar_escalado, y_azar, cv=cv, groups=dados.modelo_aleatorio, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# In[ ]:





# # Forma certa

# In[31]:


from sklearn.pipeline import Pipeline #cria um processo

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

pipeline = Pipeline([('transformador',escalador), ('estimador', modelo)])
pipeline


# In[32]:


cv = GroupKFold(n_splits=10)

modelo = SVC()
resultado = cross_validate(modelo,x_azar, y_azar, cv=cv, groups=dados.modelo_aleatorio, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)

