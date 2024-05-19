import pandas as pd
import streamlit as st
import requests
import tempfile

# URL de download direto do Google Drive
url = 'https://drive.google.com/uc?id=1-KpcABxWug0gfsnk_h3DgtWjN-3tOv8H'

# Função para baixar o arquivo do Google Drive
def download_file_from_google_drive(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=32768):
            file.write(chunk)

# Criando um arquivo temporário
with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
    # Baixando o arquivo
    download_file_from_google_drive(url, tmp_file.name)

    # Carregando os dados do arquivo Parquet
    dados = pd.read_parquet(tmp_file.name)

# Exibindo uma amostra de 10 registros no Streamlit
st.write("Amostra dos dados:")
st.dataframe(dados.sample(10))
