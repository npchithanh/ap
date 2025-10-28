import streamlit as st
from lightgbm import LGBMClassifier
import joblib
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Acute pancreatitis - PAR index", layout="wide")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "lgbm.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
@st.cache_data
def load_data():
    return joblib.load('X_train.joblib')
model = load_model()
X_train = load_data()
explainer = shap.Explainer(model = model, masker=X_train, feature_names=X_train.columns)
st.title('Acute pancreatitis - PAR index')

dic = {}

st.subheader('Clinical Index')
col1, col2 = st.columns(2)
with col1:
    dic['age'] = st.number_input('Age [19, 95]', min_value=19, max_value = 95)
    dic['resp_rate_mean'] = st.number_input('Respiratory Rate [9, 38]', min_value = 9, max_value = 38, step=1)
with col2:
    dic['mbp_mean'] = st.number_input('Mean Blood Pressure [53, 133 mmHg]', min_value=53, max_value=133, step = 1)
    dic['temperature_mean'] = st.number_input('Temperature [33.6 - 40 0c]', min_value=33.6, max_value=40.0, step=0.1, value=35.1)
st.subheader('Biochemical Index')
col1, col2= st.columns(2)
with col1:
    dic['aniongap_max'] = st.number_input('Anion Gap [7, 49]', min_value=7, max_value = 49, step=1)
    dic['lactate'] = st.number_input('Lactate [0-28 IU/L]', min_value = 0.0, max_value=28.0, step=0.1, value=15.3)
    dic['pt_max'] = st.number_input('Prothrombin Time [8.8, 150]', min_value=8.8, max_value = 150.0, step=0.1)
with col2:
    dic['PAR'] = st.number_input('PAR [0.33, 9.1]', min_value=0.33, max_value = 9.1, step=0.01, value=8.0)
    dic['rdw_max'] = rdw = st.number_input('Red Cell Distribution Width [11.8, 35]', min_value=11.8, max_value = 35.0, value=30.0)

    # Anion gap
    # Lactate
    # Prothrombin time
    # Phosphate/Albumin (PAR) ratio
    # Red Cell Distribution Width

arr = ['Survived', 'Died']

if st.button('Run'):
    df_pred = pd.DataFrame({k: [dic[k]] for k in dic})
    # # st.write(df_pred)
    # st.write(dic)
    # st.write(df_pred.iloc[[1]])
    pred = model.predict(df_pred.iloc[[0]])[0]
    prob = model.predict_proba(df_pred.iloc[[0]])
    # #st.write(pred)
    # #st.write(prob[0, pred])

    st.write(f'Predicted: {arr[pred]}')
    # Giá trị xác suất rủi ro
    risk_prob = round(prob[0, pred], 2) * 100  # phần trăm
    html_code = f"""
    <div style="display:flex; align-items:center;">
    <div style="
        position: relative;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: conic-gradient(#1E90FF {risk_prob*3.6}deg, #e6e6e6 {risk_prob*3.6}deg);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        font-weight: 600;
        color: #333;">
        <div style="
            position: absolute;
            width: 90px;
            height: 90px;
            background: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;">
            {risk_prob / 100:.2f}
        </div>
    </div>
    <div style="margin-left: 20px;">
        <div style="color:#1E90FF; font-weight:600;">Risk probs</div>
        <div style="font-size:20px; font-weight:500;">{risk_prob:.2f}%</div>
    </div>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)

    shap_values = explainer(df_pred.iloc[:1])
    #shap.plots.w
    fig = plt.figure()
    #fig, ax = plt.subplots()
    fig = shap.plots.force(shap_values[0,:], matplotlib = True, show=False)
    st.pyplot(fig)
    plt.close(fig)

    #fig = shap.plots.waterfall(shap_values[0,:,1], matplotlib = True)
    #st.pyplot(fig)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0,:], show=False)
    st.pyplot(fig)
    # # Tạo biểu đồ donut
    # fig = go.Figure(data=[go.Pie(
    #     values=[risk_prob, 100 - risk_prob],
    #     hole=0.7,
    #     textinfo='none',
    #     marker_colors=['#1E90FF', '#E0E0E0']
    # )])

    # # Thêm chữ ở giữa
    # fig.update_layout(
    #     annotations=[
    #         dict(text=f"{risk_prob/100:.2f}", x=0.5, y=0.5, font_size=24, showarrow=False, font_color="#333"),
    #     ],
    #     showlegend=False,
    #     width=100,
    #     height=100,
    #     margin=dict(l=0, r=0, t=0, b=0)
    # )

    # # Hiển thị trên Streamlit
    # col1, col2 = st.columns([1, 1.5])
    # with col1:
    #     st.plotly_chart(fig, use_container_width=True)
    # with col2:
    #     st.markdown(f"""
    #         <div style='padding-top:25px;'>
    #             <span style='color:#1E90FF; font-weight:600;'>Risk probs</span><br>
    #             <span style='font-size:20px; font-weight:500;'>{risk_prob:.2f}%</span>
    #         </div>
    #     """, unsafe_allow_html=True)
    
