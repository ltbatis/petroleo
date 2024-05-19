import pandas as pd
import streamlit as st

url = 'https://drive.google.com/uc?id=1-KpcABxWug0gfsnk_h3DgtWjN-3tOv8H'

st.write("Carregando dados...")
dados = pd.read_parquet(url)

st.write("Amostra dos dados:")
st.dataframe(dados.sample(10))
