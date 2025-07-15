import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw Data**')
  url = "https://raw.githubusercontent.com/Nijesu-alt/nijesu.machinelearning/refs/heads/master/customer_churn_dataset-testing-master.csv"
  df = pd.read_csv(url)
  df

  st.write('**X**')
  X = df.drop('Churn', axis=1)
  X
  
  st.write('**y**')
  y = df.Churn
  y


