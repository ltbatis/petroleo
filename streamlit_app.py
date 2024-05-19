import pandas as pd
import numpy as np
import streamlit as st
import streamlit_mermaid as stmd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import plotly.express as px
from sklearn.model_selection import cross_val_score


diagrama_mermaid = """
```mermaid
graph LR
    conflitos[Conflitos] -->|Processa| PySpark{{PySpark}}
    dow_jones[Dow Jones] -->|Processa| PySpark
    nasdaq[Nasdaq] -->|Processa| PySpark
    em_data_disasters[EM Data Disasters] -->|Processa| PySpark
    preco_dolar[Preço do Dólar] -->|Processa| PySpark
    preco_petroleo[Preço do Petróleo] -->|Processa| PySpark
    PySpark -->|Gera| resultado[Parquet: Resultado]
    resultado -->|Consumido por| PowerBI[Power BI]
    PowerBI -->|Integra| StreamLit[StreamLit]
    preco_petroleo -->|Direto| StreamLit
    
    classDef fontes fill:#f96,stroke:#333,stroke-width:2px;
    classDef ferramentas fill:#bbf,stroke:#333,stroke-width:2px;
    classDef dados fill:#9f6,stroke:#333,stroke-width:2px;
    classDef resultado fill:#f9f,stroke:#333,stroke-width:2px;
    
    class conflitos,dow_jones,nasdaq,em_data_disasters,preco_dolar,preco_petroleo fontes;
    class PySpark,PowerBI,StreamLit ferramentas;
    class resultado dados;
```"""

st.title("Análise e Previsão de Preço do Petróleo")
stmd.st_mermaid(diagrama_mermaid)
st.markdown("""
### Links Úteis:
- [Nosso app no Streamlit](https://petroleo-ptp3vt6vxu7u2psoipv49b.streamlit.app/)
- [Nosso report no Power BI](https://app.powerbi.com/links/SkSXFlDcuV?ctid=11dbbfe2-89b8-4549-be10-cec364e59551&pbi_source=linkShare)
- [Nosso repositório no GitHub](https://github.com/ltbatis/petroleo)
- [Nossos arquivos](https://drive.proton.me/urls/V3FRMY1X6C#WA4wi6NN8DPi)
""")

st.write("""
Bem-vindo à nossa análise e previsão de preços do petróleo. Neste projeto, buscamos explorar e aplicar diferentes técnicas de previsão de séries temporais para prever os preços futuros do petróleo. A previsão de preços de commodities, como o petróleo, é crucial para várias indústrias e setores econômicos, pois influencia decisões estratégicas, planejamento financeiro e gestão de riscos.
""")

st.write("""
### Metodologia

Inicialmente, tentamos utilizar modelos clássicos de séries temporais, como ARIMA (AutoRegressive Integrated Moving Average) e Holt-Winters (também conhecido como Smoothing Exponencial), que são amplamente usados para previsão de séries temporais devido à sua capacidade de modelar e prever dados dependentes do tempo.

#### Desafios Encontrados

1. **Estacionariedade**:
   - A primeira etapa crucial em qualquer análise de séries temporais é garantir que a série seja estacionária, ou seja, suas propriedades estatísticas (como média e variância) não mudam ao longo do tempo.
   - Realizamos várias tentativas de diferenciação dos dados para remover tendências e tornar a série estacionária. Mesmo assim, a estacionariedade completa não foi atingida.

2. **Auto ARIMA**:
   - Utilizamos a função `auto_arima` para encontrar os melhores parâmetros p, d e q automaticamente para o modelo ARIMA.
   - Apesar das otimizações, as previsões geradas foram nulas, indicando problemas subjacentes na série temporal que o modelo ARIMA não conseguiu capturar adequadamente.

3. **Holt-Winters**:
   - Tentamos também o modelo Holt-Winters para considerar a sazonalidade nos dados.
   - No entanto, as previsões continuaram a ser inadequadas, com resultados nulos.

#### Solução Adotada

Após vários testes e ajustes nos modelos tradicionais de séries temporais, decidimos optar por uma abordagem de Regressão Linear. A regressão linear é um método simples, porém poderoso, que pode modelar a relação linear entre uma variável dependente e uma variável independente.

**Por que Regressão Linear?**
- **Simplicidade e Eficácia**: A regressão linear é fácil de implementar e interpretar, fornecendo uma solução eficaz quando os modelos tradicionais falham.
- **Estabilidade nas Previsões**: A regressão linear mostrou-se mais estável nas previsões, oferecendo resultados coerentes e utilizáveis.
""")

# Carregando e preparando os dados
st.subheader("Carregamento dos Dados")
code = """
dados = pd.read_csv('data/preco_petroleo.csv')
dados['data'] = pd.to_datetime(dados['data'])
dados.set_index('data', inplace=True)
dados.sort_index(inplace=True)
st.dataframe(dados.head())
"""
st.code(code, language='python')
st.write("Neste trecho, carregamos os dados do preço do petróleo, convertendo a coluna de data para o formato datetime e configurando-a como índice. Além disso, classificamos os dados por data e exibimos as primeiras linhas do dataframe.")

dados = pd.read_csv('data/preco_petroleo.csv')
dados['data'] = pd.to_datetime(dados['data'])
dados.set_index('data', inplace=True)
dados.sort_index(inplace=True)
st.dataframe(dados.head())

# Visualização dos Preços do Petróleo ao Longo do Tempo
st.subheader("Visualização dos Preços do Petróleo ao Longo do Tempo")
code = """
fig, ax = plt.subplots()
ax.plot(dados['preco'], label='Preço do Petróleo')
ax.set_xlabel('Data')
ax.set_ylabel('Preço')
ax.legend()
st.pyplot(fig)
"""
st.code(code, language='python')
st.write("Aqui, criamos um gráfico de linha para visualizar a variação dos preços do petróleo ao longo do tempo.")

fig, ax = plt.subplots()
ax.plot(dados['preco'], label='Preço do Petróleo')
ax.set_xlabel('Data')
ax.set_ylabel('Preço')
ax.legend()
st.pyplot(fig)

# Decomposição de tendência e sazonalidade
st.subheader("Decomposição de Tendência e Sazonalidade")
model_type = st.selectbox("Modelo de decomposição:", ["additive", "multiplicative"])
period = st.number_input("Período:", min_value=1, value=12)
code = f"""
result = seasonal_decompose(dados['preco'], model='{model_type}', period={period})
fig = result.plot()
st.pyplot(fig)
"""
st.code(code, language='python')
st.write(f"Nesta seção, realizamos a decomposição da série temporal para separar a tendência, sazonalidade e resíduos. Utilizamos o modelo {model_type} com um período de {period} meses.")

result = seasonal_decompose(dados['preco'], model=model_type, period=period)
fig = result.plot()
st.pyplot(fig)

# Análise de autocorrelação
st.subheader("Análise de Autocorrelação")
code = """
fig, ax = plt.subplots()
plot_acf(dados['preco'], lags=24, ax=ax)
st.pyplot(fig)
"""
st.code(code, language='python')
st.write("Realizamos a análise de autocorrelação para verificar a correlação dos preços do petróleo com seus próprios valores defasados.")

fig, ax = plt.subplots()
plot_acf(dados['preco'], lags=24, ax=ax)
st.pyplot(fig)

st.subheader("Análise de Autocorrelação Parcial")
code = """
fig, ax = plt.subplots()
plot_pacf(dados['preco'], lags=24, ax=ax)
st.pyplot(fig)
"""
st.code(code, language='python')
st.write("Analisamos a autocorrelação parcial para entender a correlação dos preços do petróleo com suas defasagens, eliminando os efeitos das defasagens intermediárias.")

fig, ax = plt.subplots()
plot_pacf(dados['preco'], lags=24, ax=ax)
st.pyplot(fig)

# Adicionando uma coluna de tempo para regressão linear
st.subheader("Preparação dos Dados")
code = """
dados['t'] = range(len(dados))
"""
st.code(code, language='python')
st.write("Adicionamos uma coluna de tempo 't' ao dataframe para ser usada como variável independente na regressão linear.")

dados['t'] = range(len(dados))

# Modelagem
st.subheader("Modelagem")
st.write("""
Depois de enfrentar desafios com a estacionariedade e métodos tradicionais de séries temporais, optamos pela Regressão Linear para modelar os dados.
""")
code = """
X = dados[['t']]  # Variável independente
y = dados['preco']  # Variável dependente
modelo_lr = LinearRegression()
modelo_lr.fit(X, y)
"""
st.code(code, language='python')
st.write("Preparamos as variáveis independentes e dependentes para a regressão linear e ajustamos o modelo de regressão linear aos dados.")

X = dados[['t']]  # Variável independente
y = dados['preco']  # Variável dependente
modelo_lr = LinearRegression()
modelo_lr.fit(X, y)

# Avaliação do Modelo
st.subheader("Avaliação do Modelo")

# Previsões no conjunto de treino
previsoes_train = modelo_lr.predict(X)

# Calculando o RMSE
rmse = np.sqrt(mean_squared_error(y, previsoes_train))

# Calculando o MAE
mae = np.mean(np.abs(y - previsoes_train))

# Validação Cruzada
scores = cross_val_score(modelo_lr, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)

# Formatação das métricas como bullet points
st.write("""
### Avaliação do Modelo

#### Métricas de Desempenho
Para avaliar o desempenho do modelo de regressão linear, utilizamos duas métricas principais: RMSE (Root Mean Square Error) e MAE (Mean Absolute Error). Essas métricas nos ajudam a entender o quão próximo o modelo está dos valores reais.

#### Validação Cruzada
Além disso, implementamos a validação cruzada com 5 folds. A validação cruzada é uma técnica que divide os dados em partes (folds) e treina o modelo várias vezes, cada vez utilizando um conjunto diferente de dados para validação. Isso nos ajuda a garantir que o modelo não está superajustado (overfitting) aos dados de treinamento e que terá um bom desempenho em dados não vistos.

#### Resultados:
- **RMSE do modelo**: {:.2f}
- **MAE do modelo**: {:.2f}
- **RMSE médio na validação cruzada**: {:.2f}
- **Desvio padrão do RMSE na validação cruzada**: {:.2f}
""".format(rmse, mae, rmse_scores.mean(), rmse_scores.std()))

# Teste de Previsão Interativa
st.subheader("Teste de Previsão Interativa")
dias_previsao = st.number_input('Dias para prever:', min_value=1, max_value=365, value=30)
code = """
if st.button('Gerar Previsões'):
    X_pred = np.array(range(len(dados) + dias_previsao)).reshape(-1, 1)
    previsoes = modelo_lr.predict(X_pred)
    
    datas_previsoes = pd.date_range(start=dados.index[-1], periods=dias_previsao + 1, freq='D')[1:]
    previsoes_df = pd.DataFrame({'Previsão': previsoes[-dias_previsao:]}, index=datas_previsoes)
    
    fig, ax = plt.subplots()
    ax.plot(dados['preco'], label='Dados Históricos')
    ax.plot(previsoes_df, label='Previsões', color='red')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend()
    st.pyplot(fig)
    
    st.write("Previsões geradas:")
    st.dataframe(previsoes_df)
"""
st.code(code, language='python')
st.write("Aqui, configuramos uma interface interativa para gerar previsões dos preços do petróleo. O usuário pode inserir o número de dias para prever, e ao clicar no botão, as previsões são geradas e exibidas em um gráfico e tabela.")

if st.button('Gerar Previsões'):
    X_pred = np.array(range(len(dados) + dias_previsao)).reshape(-1, 1)
    previsoes = modelo_lr.predict(X_pred)
    
    datas_previsoes = pd.date_range(start=dados.index[-1], periods=dias_previsao + 1, freq='D')[1:]
    previsoes_df = pd.DataFrame({'Previsão': previsoes[-dias_previsao:]}, index=datas_previsoes)
    
    fig, ax = plt.subplots()
    ax.plot(dados['preco'], label='Dados Históricos')
    ax.plot(previsoes_df, label='Previsões', color='red')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend()
    st.pyplot(fig)
    
    st.write("Previsões geradas:")
    st.dataframe(previsoes_df)

st.write("""
### Documentação e Recursos Educativos

Para saber mais sobre os métodos e bibliotecas utilizados neste projeto, confira os links abaixo:

- [Documentação do pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Documentação do NumPy](https://numpy.org/doc/)
- [Documentação do scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Documentação do Matplotlib](https://matplotlib.org/stable/users/index.html)
- [Documentação do statsmodels](https://www.statsmodels.org/stable/)
- [Documentação do Plotly](https://plotly.com/python/)

### Notas de Rodapé e Referências

1. **Estacionariedade**: Uma série temporal é dita estacionária se suas propriedades estatísticas, como média, variância e autocorrelação, são constantes ao longo do tempo. [Saiba mais](https://en.wikipedia.org/wiki/Stationary_process)
2. **ARIMA**: Modelos AutoRegressive Integrated Moving Average são utilizados para analisar e prever séries temporais. [Saiba mais](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
3. **Holt-Winters**: Um método de suavização exponencial para previsão de séries temporais com componentes de tendência e sazonalidade. [Saiba mais](https://en.wikipedia.org/wiki/Exponential_smoothing#Triple_exponential_smoothing_(Holt-Winters))
4. **Regressão Linear**: Um método estatístico para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes. [Saiba mais](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression)

Este projeto demonstra a importância de explorar diferentes abordagens e metodologias ao trabalhar com previsões de séries temporais. Embora os métodos tradicionais de séries temporais como ARIMA e Holt-Winters sejam amplamente utilizados, eles podem não ser adequados em todos os casos. A regressão linear, apesar de sua simplicidade, provou ser uma solução eficaz para prever os preços do petróleo neste contexto. Esperamos que esta análise e ferramenta interativa sejam úteis para entender e prever as tendências futuras dos preços do petróleo, auxiliando na tomada de decisões estratégicas e financeiras.
""")

st.success("Análise concluída com sucesso!")
