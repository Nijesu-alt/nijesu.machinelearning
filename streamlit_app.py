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
  tenure = st.slider('Tenure', 1, 70, 32)
  useage_freq = st.slider('Usage Frequency', 1, 40, 15)
  sup_calls = st.slider('Support Calls', 0, 15, 6)
  pay_delay = st.slider('Payment Delay', 0, 40, 20)
  tot_spend = st.number_input('Total Spend', 100, 1000, 500, 50)
  last_int = st.slider('Last Interaction', 1, 50, 25, 1)

st.text_input('what do you think?')

data = {
  'Age' : age,
  'Gender' : gender,
  'Tenure' : tenure,
  'Usage Frequency' : useage_freq,
  'Support Calls' : sup_calls,
  'Payment Delay' : pay_delay,
  'Subscription Type' : sub_type,
  'Contract Length' : con_length,
  'Total Spend' : tot_spend,
  'Last Interaction' : last_int
}

input_df = pd.DataFrame(data, index=[0])
