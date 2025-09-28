import streamlit as st
#from lightgbm import LGBMClassifier
import joblib
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rfc.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
model = load_model()

st.title('score PAR Predict AP')

dic = {}

dic['age'] = st.slider('Age', min_value=19, max_value = 95)
dic['rdw_max'] = rdw = st.slider('RDW', min_value=11.8, max_value = 35.0)
dic['PAR'] = st.slider('PAR', min_value=0.33, max_value = 9.1, step=0.01)
dic['mbp_mean'] = st.slider('MBP', min_value=53.4, max_value = 132.3, step=0.01)
dic['temperature_mean'] = st.slider('Temperature', min_value=33.6, max_value = 40.1, step=0.01)
dic['sepsis'] = st.selectbox('Sepsis', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])
dic['vasopressin'] = st.selectbox('Vasopressin', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])
dic['crrt'] = st.selectbox('Crrt', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])

if st.button('Run'):
    df_pred = pd.DataFrame(dic)
    st.write(df_pred.iloc[[1]])
    pred = model.predict(df_pred.iloc[[0]])[0]
    prob = model.predict_proba(df_pred.iloc[[0]])
    #st.write(pred)
    #st.write(prob[0, pred])
    st.write(f'Predicted: {pred}, Probality: {prob[0, pred]:.2}')
    

# cols_sel = ['age', 'rdw_max', 'PAR', 'mbp_mean', 'temperature_mean', 'sepsis',
#         'vasopressin', 'crrt']