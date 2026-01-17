import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IDS - Détection d’attaques", layout="wide")

st.title(" IDS – Détection d’attaques réseau (NSL-KDD)")

pipeline = joblib.load("ids_pipeline.pkl")

uploaded_file = st.file_uploader(
    " Importer un fichier des logs (CSV)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des logs")
    st.dataframe(data.head())

    X = data.drop(columns=["label", "difficulty"], errors="ignore")

    predictions = pipeline.predict(X)
    data["prediction"] = predictions
    data["prediction"] = data["prediction"].map({0: "NORMAL", 1: "ATTAQUE"})

    st.subheader(" Résultats de détection")
    st.dataframe(data[["prediction"]].value_counts().rename("Nombre"))

    st.subheader(" Détails des attaques détectées")
    st.dataframe(data[data["prediction"] == "ATTAQUE"])

    st.success(" Analyse terminée")