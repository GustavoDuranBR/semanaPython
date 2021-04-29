import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# gerar graficos
tabela = pd.read_csv("advertising.csv")
sns.pairplot(tabela)
plt.show()
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
plt.show()

# iniciar machine learning
x = tabela.drop("Vendas", axis=1)
y = tabela["Vendas"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

linear = LinearRegression()
randomforest = RandomForestRegressor()

linear.fit(x_treino, y_treino)
randomforest.fit(x_treino, y_treino)

teste_linear = linear.predict(x_teste)
teste_random = randomforest.predict(x_teste)

# r2
r2_linear = metrics.r2_score(y_teste, teste_linear)
r2_ramdom = metrics.r2_score(y_teste, teste_random)
print(r2_linear, r2_ramdom)

# erro
erro_linear = metrics.mean_squared_error(y_teste, teste_linear)
erro_random = metrics.mean_squared_error(y_teste, teste_random)
print(erro_linear, erro_random)

# visualização gráfica das previsões
tabela_comparacao = pd.DataFrame()
tabela_comparacao["Vendas Reais"] = y_teste
tabela_comparacao["Previsão Random"] = teste_random
tabela_comparacao = tabela_comparacao.reset_index(drop=True)

print(tabela_comparacao)
sns.lineplot(data=tabela_comparacao)
plt.show()
