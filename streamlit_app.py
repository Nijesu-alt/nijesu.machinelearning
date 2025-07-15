import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model')

url = "https://raw.githubusercontent.com/nijesu/machinelearning/main/data/customer_churn.csv"
df = pd.read_csv(url)
df
