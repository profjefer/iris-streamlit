import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

# --- Configuração da página ---
st.set_page_config(page_title="Data Science App", layout="wide") # pagina em paisagem

st.title("🔬 Explorador de Dados - Iris Dataset") #titulo
st.markdown("Um exemplo simples de app de Data Science com Streamlit.")

# --- Carregando dados ---
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [*iris.feature_names, "target"]
    df["especie"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df

df = load_data()

# --- Sidebar com filtros ---
st.sidebar.header("⚙️ Filtros")
especies = st.sidebar.multiselect(
    "Selecione as espécies:",
    options=df["especie"].unique(),
    default=df["especie"].unique()
)

df_filtrado = df[df["especie"].isin(especies)]

# --- Métricas ---
col1, col2, col3 = st.columns(3)
col1.metric("Total de registros", len(df_filtrado))
col2.metric("Espécies selecionadas", len(especies))
col3.metric("Features", df_filtrado.shape[1] - 2)

st.divider()

# --- Tabela de dados ---
with st.expander("📋 Ver dados brutos"):
    st.dataframe(df_filtrado, use_container_width=True)

# --- Gráficos ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Dispersão")
    eixo_x = st.selectbox("Eixo X", df.columns[:-2], key="x")
    eixo_y = st.selectbox("Eixo Y", df.columns[:-2], index=1, key="y")
    fig = px.scatter(df_filtrado, x=eixo_x, y=eixo_y, color="especie")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Distribuição")
    coluna = st.selectbox("Variável", df.columns[:-2])
    fig2 = px.histogram(df_filtrado, x=coluna, color="especie", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)

# --- Estatísticas descritivas ---
st.subheader("📊 Estatísticas Descritivas")
st.dataframe(df_filtrado.describe(), use_container_width=True)