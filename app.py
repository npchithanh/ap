import streamlit as st
#from lightgbm import LGBMClassifier
import joblib
import pandas as pd
import os
import joblib
st.set_page_config(page_title="Acute pancreatitis - PAR index", layout="wide")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "lgbm.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
model = load_model()

st.title('Acute pancreatitis - PAR index')

dic = {}
col1, col2, col3 = st.columns(3)
with col1:
    dic['age'] = st.number_input('Age [19, 95]', min_value=19, max_value = 95)
    dic['aniongap_max'] = st.number_input('Anion Gap [7, 49]', min_value=7, max_value = 49, step=1)
    dic['lactate'] = st.number_input('Lactate [0-28 IU/L]', min_value = 0.0, max_value=28.0, step=0.1)
with col2:  
    dic['rdw_max'] = rdw = st.number_input('Red Cell Distribution Width [11.8, 35]', min_value=11.8, max_value = 35.0)
    dic['PAR'] = st.number_input('PAR [0.33, 9.1]', min_value=0.33, max_value = 9.1, step=0.01)
    dic['pt_max'] = st.number_input('Prothrombin Time [8.8, 150]', min_value=8.8, max_value = 150.0, step=0.1)
with col3:
    dic['resp_rate_mean'] = st.number_input('Respiratory Rate [9.6, 38.2]', min_value = 9.6, max_value = 38.2, step=0.1)
    dic['mbp_mean'] = st.number_input('Mean Blood Pressure [53.4, 132.3 IU/L]', min_value=53.4, max_value=132.3, step=0.1)
    dic['temperature_mean'] = st.number_input('Temperature [33.6 - 40 0c]', min_value=33.6, max_value=40.0, step=0.1)

if st.button('Run'):
    df_pred = pd.DataFrame({k: [dic[k]] for k in dic})
    # # st.write(df_pred)
    # st.write(dic)
    # st.write(df_pred.iloc[[1]])
    pred = model.predict(df_pred.iloc[[0]])[0]
    prob = model.predict_proba(df_pred.iloc[[0]])
    # #st.write(pred)
    # #st.write(prob[0, pred])
    st.write(f'Predicted: {pred}, Probality: {prob[0, pred]:.2}')
    
# 'age', 'pt_max', 'aniongap_max', 'rdw_max', 'lactate', 'PAR',
#        'resp_rate_mean', 'mbp_mean', 'temperature_mean'
# age                 19.000000
# pt_max               8.800000
# aniongap_max         7.000000
# rdw_max             11.800000
# lactate             -0.011744
# PAR                  0.333333
# resp_rate_mean       9.653846
# mbp_mean            53.400000
# temperature_mean    33.600000
# dtype: float64
# age                  95.000000
# pt_max              150.000000
# aniongap_max         49.000000
# rdw_max              34.900000
# lactate              28.000000
# PAR                   9.076923
# resp_rate_mean       38.173913
# mbp_mean            132.272727
# temperature_mean     40.104118