# Projeto Regressão

A ideia do projeto é testar algoritmos de ML de regressão a fim de determinar o custo médico de famílias, dadas as variáveis abaixo:

- age: idade do beneficiário principal.

- sex: sexo do contratante do seguro (feminino ou masculino).

- bmi: índice de massa corporal (kg/m ^ 2), idealmente 18,5 a 24,9.

- children: Número de filhos cobertos pelo seguro saúde / Número de dependentes.

- smoker: Fumante ou não-fumante.

- region: área residencial do beneficiário nos EUA, nordeste, sudeste, sudoeste, noroeste.

- charges: custos médicos individuais faturados pelo seguro de saúde.

Sendo a variável charges nosso alvo.

O dataset escolhido foi o insurance.csv, do Kaggle  <a href="https://www.kaggle.com/datasets/mirichoi0218/insurance">Kaggle Insurance</a>

Os algoritmos de regressão testados serão: KNeighborsRegressor, SVR, LinearRegression, DecisionTreeRegressor, RandomForestRegressor.

Carregamos todas as biliotecas necessárias para início de projetos e carrega-se o dataset:

```
import pandas as pd
import numpy as np

#Carregamento do dataset
df = pd.read_csv('./dataset/insurance.csv')
```

Separando as variáveis preditivas/alvo e realizando a separação entre treino e teste (validação):

```
X = df.iloc[:, [0,1,2,3,4,5]]
y = df.iloc[:, 6]

# Separação entre teste e treino
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.30, random_state=99)
```

Antes de realizar o pré-processamento de fato, vamos dar uma olhada na correlação das variáveis entre si. Na Figura abaixo é possível ver tais correlações,
Não existe correlações fortes entre nenhuma coluna, logo não é necessário eliminar nenhuma delas por estarem representando outras.

<div align="center">
  <img src="https://user-images.githubusercontent.com/102380417/186486475-765dc718-234f-48c9-84b6-9500e0238d6d.png" width="700px" />
</div>

Nosso dataset possui três colunas com variáveis categóricas nominais ('sex','smoker','region'). Para transformá-las, utilizei o ColumnTransformer com o OneHotEncoder.
Para as variáveis numéricas ('age', 'bmi', 'children') utilizei os métodos StandardScaler e MinMaxScaler para fins de comparação. A métrica de avaliação
para este projeto vai ser a raíz do erro quarático médio (RMSE).
Para organizar os passos do pré-processamento e o treinamento em si, utilizei o Pipeline, abaixo está o código para obter os valores do MSE e RMSE.

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


num_features = ['age', 'bmi', 'children']
cat_features = ['sex','smoker','region']
modelos = [KNeighborsRegressor(), SVR(), LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
esc = [MinMaxScaler(), StandardScaler()]

for scaler in esc:
    print(f'PARA {scaler}:\n')
    transformer = ColumnTransformer(transformers=[
        ('transf2', OneHotEncoder(), cat_features),
        ('scaler', scaler, num_features)
    ],remainder='passthrough')
    
    for modelo in modelos:
        pipe = Pipeline(steps=[
            ('transformer', transformer),
            ('algorithm', modelo)
        ])
    
        pipe.fit(X_treino, y_treino)
        kfold = KFold(n_splits = 10, shuffle=True, random_state=99)
        results = cross_val_score(pipe, X_treino, y_treino, cv=kfold, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-results)
        validation = rmse.mean()
        print(f'Algoritmo: {modelo}')
        print('Validação Cruzada: %.3f' %validation)
        print('----------------------------')
```
O resultado do RMSE para cada algoritmo pode ser visto abaixo:

```
PARA MinMaxScaler():

Algoritmo: KNeighborsRegressor()
Validação Cruzada: 5925.302
----------------------------
Algoritmo: SVR()
Validação Cruzada: 12767.717
----------------------------
Algoritmo: LinearRegression()
Validação Cruzada: 6020.218
----------------------------
Algoritmo: DecisionTreeRegressor()
Validação Cruzada: 6357.274
----------------------------
Algoritmo: RandomForestRegressor()
Validação Cruzada: 4663.212
----------------------------

PARA StandardScaler():

Algoritmo: KNeighborsRegressor()
Validação Cruzada: 6084.345
----------------------------
Algoritmo: SVR()
Validação Cruzada: 12778.002
----------------------------
Algoritmo: LinearRegression()
Validação Cruzada: 5978.024
----------------------------
Algoritmo: DecisionTreeRegressor()
Validação Cruzada: 6277.922
----------------------------
Algoritmo: RandomForestRegressor()
Validação Cruzada: 4627.041
----------------------------

```
Tanto para a padronização (StandardScaler) como para a normalização (MinMaxScaler) o algoritmo que obteve o menor RMSE foi o Random Forest.
E dentre os dois, o menos valor de RMSE foi com a padronização, logo, este algoritmo com a padronização vai ser o utilizado para receber os valores
de teste.

```
from sklearn.metrics import mean_squared_error
# Predição dos valores de teste
prev_random = pipe.predict(X_teste)
print('RMSE = %.2f' %np.sqrt(mean_squared_error(y_teste, prev_random)))

RMSE = 5100.02
```

A respos




