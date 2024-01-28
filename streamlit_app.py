import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Título do aplicativo
st.title('Projeção do Valor do Petróleo Brent')

# Breve descrição do trabalho
st.write('Este aplicativo realiza projeções do valor do petróleo usando o modelo Prophet de séries temporais.')
st.header('Base de dados para projeção')


# Função para fazer o web scraping
@st.cache_data
def fazer_web_scraping(url):
    dados = pd.read_html(url, thousands='.', decimal=',', parse_dates=True)[2][1:]
    dados.columns = ['Date', 'Price']
    dados['Price'] = dados['Price'].astype(float)
    dados['Date'] = pd.to_datetime(dados['Date'], dayfirst=True)
    dados = dados.sort_values(by='Date', ascending=True)
    dados = dados.reset_index(drop=True)
    dados.to_csv('raw_data.csv', index=False)
    return dados

# Bloco try-except para a tentativa de web scraping
try:
    # Tentativa de web scraping
    raw_data = fazer_web_scraping('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view')
    st.success('Dados carregados com sucesso!')

except Exception as e:
    st.error(f'Falha ao carregar dados: {e}')

# Opção para baixar a base de dados (dentro de um bloco condicional para verificar se os dados existem)
if 'raw_data' in locals():
    st.download_button(
        label="Baixar dados em CSV",
        data=raw_data.to_csv(index=False),
        file_name='raw_data.csv',
        mime='text/csv',
    )


# Função para carregar dados de um arquivo CSV
@st.cache_data
def carregar_dados(nome_arquivo):
    dados = pd.read_csv(nome_arquivo)
    return dados

# Carregar e exibir informações da base de dados
data = carregar_dados('raw_data.csv')

# Encontrar a data mínima e máxima
data_min = data['Date'].min()
data_max = data['Date'].max()
st.markdown(f'**Período disponível na base de dados:** {data_min} até {data_max}')


# Projeção de valores futuros
st.header('Projeção de Valores Futuros')

# Preparando os dados para o modelo Prophet
data['ds'] = pd.to_datetime(data['Date'])
data['y'] = data['Price']

# Ajustando o modelo Prophet
@st.cache_resource
def ajustar_modelo(data):
    model = Prophet()
    model.fit(data)
    return model

model = ajustar_modelo(data)

# Campo para o usuário definir o número de dias para projetar
dias_para_projecao = st.number_input('Digite o número de dias para projeção:', min_value=1, max_value=30, value=5)

# Criando um DataFrame para previsões
future = model.make_future_dataframe(periods=dias_para_projecao)
forecast = model.predict(future)

# Campo para o usuário definir o número de dias para visualização dos dados
dias_historicos = st.number_input('Digite o número de dias históricos para visualizar no gráfico:', 
                                  min_value=1, max_value=len(data), value=30)

# Plotando o gráfico com projeções
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['ds'].tail(dias_historicos), data['y'].tail(dias_historicos), label='Dados Históricos')
ax.plot(forecast['ds'].tail(dias_para_projecao), forecast['yhat'].tail(dias_para_projecao), label='Projeções', color='red')
ax.fill_between(forecast['ds'].tail(dias_para_projecao), forecast['yhat_lower'].tail(dias_para_projecao), forecast['yhat_upper'].tail(dias_para_projecao), color='red', alpha=0.2)
ax.set_title('Projeção de preço por barril do petróleo bruto Brent (FOB)')
ax.set_xlabel('Data')
ax.set_ylabel('Preço em USD')
plt.legend()
st.pyplot(fig)

# Exibindo a tabela com projeções
st.write('Valores projetados')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias_para_projecao))

# Lista de referências utilizadas
st.markdown("""
### Referências            
- IPEA: [Preço por barril do petróleo bruto Brent (FOB)](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)
- Prophet: [Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
""")

# Informações de contato
st.markdown("""
### Contato
- **Nome:** Leonardo Augusto Thomas
- **LinkedIn:** [Perfil no LinkedIn](https://www.linkedin.com/in/leonardothomas/)
""")
