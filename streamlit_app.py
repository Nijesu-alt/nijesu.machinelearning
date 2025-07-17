import streamlit as st
import pandas as pd
import pickle 
import gzip
import requests


st.title("🤝 Customer Churn Prediction App")

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
  st.sidebar.markdown("<h3 style='color:#F63366;'>Customer Profile</h3>", unsafe_allow_html=True)
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

sub_type_map = {'Basic' : 1, 'Standard':2, 'Premium':3}
con_length_map = {'Monthly':1, 'Annual':2, 'Quarterly':3}
sub = sub_type
con = con_length
sub = sub_type_map[sub]
con = con_length_map[con]

input_df = pd.DataFrame(data, index=[0])
input_tot = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Inputed Customer Churn Features**')
  input_df

df_numeric = input_df[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']]
df_cat = input_df[['Subscription Type', 'Contract Length']]
df_cat['Subscription Type'] = sub
df_cat['Contract Length'] = con

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('gender_cols.pkl', 'rb') as f:
    gender_columns = pickle.load(f)

gender_dummies = pd.get_dummies(input_df['Gender'])
gender_dummies = gender_dummies.reindex(columns=gender_columns, fill_value=0)
gender_columns.remove('Male')
insert_pos = gender_columns.index('Female')
gender_columns.insert(insert_pos, 'Male')
gender_dummies = gender_dummies[gender_columns]



df_new = pd.concat([df_numeric, df_cat, gender_dummies], axis=1)
scaler_input = scaler.transform(df_new)
safe_input = scaler_input.tolist() if hasattr(scaler_input, "tolist") else scaler_input

with gzip.open('mymodel1.pkl.gz', 'rb') as f:
    model = pickle.load(f)

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:5000/predict", json={"features": safe_input[0]})

    if response.status_code == 200:
        result = response.json()['prediction']
        proba = response.json().get('probability', None)  # Optional: model probability

        if result == 1:
            st.error("📉 Prediction: Customer is **likely to churn**.")
            st.info("💡 Action: Offer a personalized retention incentive or engage customer support.")
        else:
            st.success("📈 Prediction: Customer is **not likely to churn**.")
            st.info("✅ Action: Continue regular engagement and monitor satisfaction.")

        if proba is not None:
            st.write(f"📊 Estimated probability of churn: **{proba:.2%}**")
    else:
        st.warning("🔒 Please log in through the Flask app first.")
    # prediction = model.predict(scaler_input)
    # proba = model.predict_proba(scaler_input)[0][1]

    # st.subheader("📊 Prediction Result")
    # st.metric(label="Churn Probability (%)", value=f"{round(proba * 100, 2)}")


    # if prediction[0] == 1:
    #   st.error("⚠️ This customer is likely to churn")
    #   st.info("💡 Consider offering loyalty incentives or personalized outreach.")

    # else:
    #   st.success("✅ This customer is likely to stay")
    #   st.info("🏆 Maintain engagement through targeted content and value-added services.")




# if input_df['Gender']=='Male'
#   male = 1.069719
#   female = -1.069719
# elif input_df['Gender'] == 'Female':
#   male = -0.934825
#   female = 0.934825
