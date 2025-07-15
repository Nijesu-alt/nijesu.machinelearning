import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model')

df = pd.read_csv('customer_churn_dataset-testing-master')
df
