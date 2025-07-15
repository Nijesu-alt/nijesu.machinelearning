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

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='CustomerID', y='Total Spend', color='Churn')

with st.sidebar:
  st.header('Input Features')
  gender = st.selectbox('Gender', ['Female', 'Male'])
  sub_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
  con_length = st.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annual'])
  age = st.slider('Age', 0, 100, 50)

st.text_input('what do you think?')


