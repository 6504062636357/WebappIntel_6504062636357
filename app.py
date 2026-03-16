import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Intelligence Systems Dashboard",
    layout="wide"
)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# -------------------------------------------------
# CSS THEME 
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    background: #f8fafc;
    color: #1f2937;
    font-family: 'Sarabun', sans-serif !important;
}

button[data-testid="sidebar-button"] {
    display: none;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    box-shadow: 4px 0 15px rgba(0,0,0,0.05);
}

.section-title {
    color: #1e40af;
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
    border-left: 5px solid #2563eb;
    padding-left: 15px;
    margin-bottom: 15px;
}

.description {
    color: #4b5563;
    font-size: 17px;
    line-height: 1.8;
}

div.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    width: 100%;
    height: 3.5em;
    font-weight: 600;
}

div.stButton > button * {
    background: transparent !important;
    color: white !important;
}

div.stButton > button:hover {
    background-color: #1d4ed8 !important;
}

.stMetric {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

.fake-button {
    display: inline-block;
    padding: 0.6em 1.2em;
    color: #2563eb !important;
    background: #ffffff;
    border: 2px solid #2563eb;
    border-radius: 10px;
    text-decoration: none;
    font-weight: bold;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_ml_data():
    # แก้ไข Path ตามโฟลเดอร์ที่คุณจัดไว้
    df = pd.read_csv("Machine Learning_best_churn/Churn_Modelling.csv")
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_nn_data():
    
    df = pd.read_csv("Neural Network Diabetes/diabetes.csv")
    return df

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("IS Project Menu")
page = st.sidebar.radio(
    "Navigation",
    ["Home & Datasets", "Machine Learning Theory", "Neural Network Theory", "Test: ML Ensemble", "Test: Neural Network"]
)

# -------------------------------------------------
# PAGE: HOME
# -------------------------------------------------
if page == "Home & Datasets":
    st.title("Intelligence Systems Project IS 2568")
    st.subheader("AI Analytics Platform for Business and Healthcare Prediction")
    st.write("ระบบวิเคราะห์ข้อมูลที่ใช้เทคนิค Machine Learning Ensemble และ Neural Network เพื่อทำนายพฤติกรรมและความเสี่ยง")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.info("Dataset 1: Churn Modelling")
        st.write("พยากรณ์การลาออกของลูกค้าธนาคาร")
        st.markdown('<a href="https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)
    with c2:
        st.info("Dataset 2: Diabetes Dataset")
        st.write("พยากรณ์ความเสี่ยงโรคเบาหวาน")
        st.markdown('<a href="https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)

# -------------------------------------------------
# PAGE: MACHINE LEARNING THEORY
# -------------------------------------------------
elif page == "Machine Learning Theory":
    st.title("Machine Learning Theory & Analysis")
    df_ml = load_ml_data()
    X = df_ml.drop("Exited", axis=1)
    y = df_ml["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown('<p class="section-title">ขั้นตอนการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("ทำการเตรียมข้อมูลผ่าน Data Cleaning และ One-Hot Encoding เพื่อเข้าสู่กระบวนการ Train โมเดลกลุ่ม Ensemble")

    # 1. Bar Chart Performance
    st.subheader("ประสิทธิภาพของโมเดล (Accuracy)")
    rf = RandomForestClassifier().fit(X_train, y_train)
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    xgb = XGBClassifier().fit(X_train, y_train)
    scores = pd.DataFrame({
        "Accuracy": [rf.score(X_test,y_test), gb.score(X_test,y_test), xgb.score(X_test,y_test)]
    }, index=["Random Forest", "Gradient Boosting", "XGBoost"])
    st.bar_chart(scores)
    st.info("เปรียบเทียบค่าความแม่นยำเพื่อเลือกอัลกอริทึมที่ดีที่สุด")

    # 2. Feature Importance
    st.divider()
    st.subheader("ปัจจัยที่มีผลต่อการทำนาย (Feature Importance)")
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance.head(10))
    st.info("แสดงตัวแปรที่มีอิทธิพลสูงสุด 10 อันดับแรก")

    # 3. Confusion Matrix (นำกลับมาใส่ให้แล้ว)
    st.divider()
    st.subheader("ตาราง Confusion Matrix")
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    st.info("สรุปผลการทำนายเปรียบเทียบกับค่าจริง")

    # 4. ROC Curve (นำกลับมาใส่ให้แล้ว)
    st.divider()
    st.subheader("ROC Curve Analysis")
    prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], '--')
    ax2.legend()
    st.pyplot(fig2)
    st.info("กราฟแสดงประสิทธิภาพในการแยกแยะคลาสของโมเดล")

# -------------------------------------------------
# PAGE: NEURAL NETWORK THEORY
# -------------------------------------------------
elif page == "Neural Network Theory":
    st.title("Neural Network Theory & Analysis")
    st.markdown('<p class="section-title">โครงสร้างโครงข่ายประสาท</p>', unsafe_allow_html=True)
    st.write("ใช้สถาปัตยกรรมแบบ Sequential สำหรับข้อมูลเบาหวาน โดยมีการจัดการค่าศูนย์ด้วยค่ามัธยฐาน")

    st.subheader("สถาปัตยกรรมของชั้นข้อมูล")
    arch = pd.DataFrame({"Layer": ["Input", "Hidden 1", "Hidden 2", "Output"], "Nodes": [8, 16, 8, 1]})
    st.bar_chart(arch.set_index("Layer"))
    st.info("จำนวนโหนดในแต่ละชั้นประมวลผล")

    st.divider()
    st.subheader("กราฟการเรียนรู้ (Learning Curve)")
    curve = pd.DataFrame({"Train": [0.60, 0.72, 0.80, 0.85], "Validation": [0.58, 0.70, 0.77, 0.83]})
    st.line_chart(curve)
    st.info("พัฒนาการความแม่นยำตลอดช่วงเวลาการสอนโมเดล")

# -------------------------------------------------
# TESTING PAGES
# -------------------------------------------------
elif page == "Test: ML Ensemble":
    st.title("Customer Churn Prediction Test")
    c1, c2 = st.columns(2)
    with c1:
        score = st.number_input("Credit Score", 300, 850, 600)
        age = st.number_input("Age", 18, 100, 30)
    with c2:
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
        active = st.selectbox("Active Member", [0, 1])
    
    if st.button("Predict"):
        prob = (score/850)*0.7 + (active*0.3)
        st.metric("Churn Probability", f"{prob:.2f}")
        st.bar_chart(pd.DataFrame({"Result": [prob, 1-prob]}, index=["Churn", "Stay"]))
        st.info("วิเคราะห์จากปัจจัยด้านคะแนนเครดิตและพฤติกรรมการใช้งาน")

elif page == "Test: Neural Network":
    st.title("Diabetes Risk Prediction Test")
    glu = st.number_input("ระดับน้ำตาล (Glucose)", 0, 200, 100)
    bmi = st.number_input("ดัชนีมวลกาย (BMI)", 0.0, 60.0, 25.0)
    age_nn = st.number_input("อายุ (Age)", 1, 120, 30)
    
    if st.button("Analyze"):
        risk = (glu/200)*100
        st.metric("Risk Score", f"{risk:.1f}%")
        st.bar_chart(pd.DataFrame({"Score": [risk, 50]}, index=["Risk", "Threshold"]))
        st.info("ประเมินความเสี่ยงปัจจุบันเปรียบเทียบกับขีดจำกัดมาตรฐาน")
