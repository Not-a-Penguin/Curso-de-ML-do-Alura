import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv("Wine.csv")

df = df.rename(columns={'Alcohol': 'Alcool', 'Ash': 'Po', 'Ash_Alcanity': 'Alcalinidade_po', 'Magnesium': 'Magnesio', 'Color_Intensity': 'Intensidade_de_cor'})

#visualizar os dados
print(df.head())
print(df.describe())

#criar matriz de correlação
matriz_corr = df.corr()