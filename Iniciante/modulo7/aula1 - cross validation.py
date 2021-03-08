#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split #importa o método para separação automática
from sklearn.tree import DecisionTreeClassifier # usa o estimador DecisionTreeClassifier de classificação e que permite análise das tomadas de decisão
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler #importa classe StandardScaler que faz uma nova escala igual para os nossos dados 
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz #módulo export_graphviz traz as informações das  tomadas de decisão
from datetime import datetime
import graphviz  #biblioteca para plotar o gráfico


# In[3]:


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


# In[4]:


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


# In[5]:


SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)
modelo = DecisionTreeClassifier(max_depth=3) #inicializa a árvore de decisão com o parâmetro de "profundidade" em 3, ou seja, três camadas


# In[6]:


raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(      x, #dados a serem analisados
                                                              y, #classes para alocar os dados
                                                              #random_state = SEED, #parâmetro para diminuir a aleatoriedade do método
                                                              test_size = 0.25, #determina a quantidade de elementos do teste em relação ao total (ou seja 25% do total)
                                                              stratify = y # parâmetro para acertar a proporcionalidade da separação de 
                                                               )
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x),len(raw_teste_x)))

modelo.fit(raw_treino_x, treino_y) #faz o modelo treinar com os dados cru de treino
previsoes = modelo.predict(raw_teste_x) #testa o modelo com os dados de teste_X   

acuracia = accuracy_score(teste_y, previsoes)*100 #verifica a acurácia com os resultados esperados e os conseguidos na previsão
print("Acurácia do modelo = %.2f%%" % acuracia) 


# In[35]:


def imprime_resultados(resultados):
    media = resultado['test_score'].mean()
    dp = resultado['test_score'].std()
    print("Acurácia média: %.2f%%" % (media*100))
    print("Intervalo de Acurácia: %.2f - %.2f%%" % ((media-2*dp)*100, (media+2*dp)*100))


# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = KFold(n_splits=10)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x, y, cv=cv, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# # Embaralhar antes de separar

# In[38]:


SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = KFold(n_splits=10, shuffle = True)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x, y, cv=cv,return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# # Simular situação horrível de azar
# pode ser azar ou um agrupamento de desbalanceado de dados

# In[50]:


dados_azar = dados.sort_values("vendido", ascending=True)
x_azar = dados_azar[["preco","idade_do_carro","km_por_ano"]]
y_azar =  dados_azar["vendido"]


# In[51]:


from sklearn.model_selection import KFold

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = KFold(n_splits=10)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x_azar, y_azar, cv=cv, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# In[56]:


SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = KFold(n_splits=10, shuffle=True)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x_azar, y_azar, cv=cv, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)


# In[57]:


from sklearn.model_selection import StratifiedKFold

SEED = 5 #número para diminuir a aleatoriedade da separação 
np.random.seed(SEED)

cv = StratifiedKFold(n_splits=10, shuffle=True)

modelo = DecisionTreeClassifier(max_depth=2)
resultado = cross_validate(modelo,x_azar, y_azar, cv=cv, return_train_score=False) # cv = 5 a 10 é a melhor opção
imprime_resultados(resultado)

