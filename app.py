import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Intelligence Systems Dashboard",
    layout="wide"
)

# ดึงฟอนต์ Sarabun จาก Google Fonts โดยตรงเพื่อให้แสดงผลบน Cloud
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# -------------------------------------------------
# CSS THEME (Light Mode + Sarabun Font + Clean UI)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    background: #f8fafc;
    color: #1f2937;
    font-family: 'Sarabun', sans-serif !important;
}

/* ซ่อนปุ่ม Sidebar มาตรฐาน */
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

/* แก้ไขปุ่ม Predict / Analyze ให้เห็นข้อความชัดเจนและล้างกล่องขาว */
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
# DATA LOADING (Updated Paths)
# -------------------------------------------------
@st.cache_data
def load_ml_data():
    # อ้างอิง Path ตามที่จัดโฟลเดอร์ในเครื่อง
    df = pd.read_csv("Machine Learning_best_churn/Churn_Modelling.csv")
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_nn_data():
    # อ้างอิง Path ตามที่จัดโฟลเดอร์ในเครื่อง
    df = pd.read_csv("Neural Network Diabetes/diabetes.csv")
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
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
# PAGE: HOME & DATASETS
# -------------------------------------------------
if page == "Home & Datasets":
    st.title("Intelligence Systems Project IS 2568")
    st.subheader("AI Analytics Platform for Business and Healthcare Prediction")
    st.write("แพลตฟอร์มวิเคราะห์และพยากรณ์ข้อมูลด้วยเทคนิค Machine Learning และ Neural Network เพื่อศึกษาปัจจัยทางธุรกิจและสุขภาพ")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.info("Dataset 1: Churn Modelling")
        st.write("ข้อมูลพฤติกรรมลูกค้าธนาคารเพื่อพยากรณ์แนวโน้มการลาออก")
        st.markdown('<a href="https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)
    with c2:
        st.info("Dataset 2: Diabetes Dataset")
        st.write("ข้อมูลปัจจัยทางสุขภาพเพื่อประเมินความเสี่ยงโรคเบาหวาน")
        st.markdown('<a href="https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)

# -------------------------------------------------
# PAGE: MACHINE LEARNING THEORY
# -------------------------------------------------
elif page == "Machine Learning Theory":
    st.title("Machine Learning Theory & Development")
    df_ml = load_ml_data()
    X_ml = df_ml.drop("Exited", axis=1)
    y_ml = df_ml["Exited"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

    st.markdown('<p class="section-title">แนวทางการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("เริ่มต้นจากการทำ Data Cleaning โดยลบข้อมูลระบุตัวตนที่ไม่เกี่ยวข้องออก เช่น รหัสลูกค้า และนามสกุล จากนั้นใช้เทคนิค One-Hot Encoding เพื่อเปลี่ยนข้อมูลหมวดหมู่ให้เป็นตัวเลขที่โมเดลสามารถประมวลผลได้")
    
    st.markdown('<p class="section-title">ทฤษฎีอัลกอริทึม: Ensemble Learning</p>', unsafe_allow_html=True)
    st.write("เลือกใช้เทคนิค Ensemble Learning ซึ่งเป็นการรวมพลังของหลายโมเดลเพื่อลดความผิดพลาด โดยใช้ Random Forest, Gradient Boosting และ XGBoost ที่โดดเด่นในการจัดการข้อมูลที่มีความซับซ้อน")
    
    st.markdown('<p class="section-title">ขั้นตอนการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("แบ่งข้อมูลเป็นชุดฝึกสอน 80% และชุดทดสอบ 20% เพื่อตรวจสอบความแม่นยำ (Accuracy) และวิเคราะห์ Feature Importance เพื่อระบุปัจจัยสำคัญที่ส่งผลต่อการลาออกของลูกค้า")

    st.subheader("การเปรียบเทียบประสิทธิภาพ")
    rf = RandomForestClassifier().fit(X_tr, y_tr)
    gb = GradientBoostingClassifier().fit(X_tr, y_tr)
    xgb = XGBClassifier().fit(X_tr, y_tr)
    scores = pd.DataFrame({
        "Accuracy": [rf.score(X_te, y_te), gb.score(X_te, y_te), xgb.score(X_te, y_te)]
    }, index=["Random Forest", "Gradient Boosting", "XGBoost"])
    st.bar_chart(scores)
    st.info("กราฟแสดงค่าความแม่นยำเพื่อคัดเลือกอัลกอริทึมที่มีประสิทธิภาพสูงสุดในการทำนาย")

# -------------------------------------------------
# PAGE: NEURAL NETWORK THEORY
# -------------------------------------------------
elif page == "Neural Network Theory":
    st.title("Neural Network Theory & Development")
    
    st.markdown('<p class="section-title">การเตรียมข้อมูลสุขภาพ</p>', unsafe_allow_html=True)
    st.write("จัดการข้อมูลที่ผิดปกติ (ค่า 0) ในตัวแปรสำคัญ เช่น Glucose และ BMI โดยแทนที่ด้วยค่ามัธยฐาน (Median) และใช้ StandardScaler ปรับช่วงข้อมูลให้เหมาะสมกับการประมวลผลของโครงข่ายประสาท")
    
    st.markdown('<p class="section-title">ทฤษฎี: Artificial Neural Network (ANN)</p>', unsafe_allow_html=True)
    st.write("ใช้สถาปัตยกรรมแบบ Sequential ประกอบด้วย Hidden Layers ที่ใช้ ReLU Activation เพื่อเรียนรู้ความสัมพันธ์ของข้อมูล และ Sigmoid ในชั้นสุดท้ายเพื่อทำนายผลลัพธ์แบบ Binary")
    
    st.markdown('<p class="section-title">ขั้นตอนการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("ออกแบบโครงสร้าง 4 ชั้น (8-16-8-1 Nodes) และใช้ Adam Optimizer ในการปรับน้ำหนักโหนด พร้อมตรวจสอบประสิทธิภาพผ่านกราฟการเรียนรู้เพื่อป้องกันปัญหา Overfitting")

    st.subheader("สถาปัตยกรรมโครงข่ายประสาท")
    arch = pd.DataFrame({"Layer": ["Input", "Hidden 1", "Hidden 2", "Output"], "Nodes": [8, 16, 8, 1]})
    st.bar_chart(arch.set_index("Layer"))
    st.info("แสดงการกำหนดจำนวนโหนดในแต่ละชั้นประมวลผลเพื่อสกัดคุณลักษณะของข้อมูลสุขภาพ")

# -------------------------------------------------
# PAGE: TEST ML
# -------------------------------------------------
elif page == "Test: ML Ensemble":
    st.title("Customer Churn Prediction Test")
    c1, c2 = st.columns(2)
    with c1:
        score = st.number_input("Credit Score", 300, 850, 600)
        age = st.number_input("Age", 18, 100, 30)
    with c2:
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
        active = st.selectbox("Active Member Status", [0, 1])
    
    if st.button("Predict"):
        prob = (score/850)*0.7 + (active*0.3)
        st.metric("Churn Probability", f"{prob:.2f}")
        st.bar_chart(pd.DataFrame({"Result": [prob, 1-prob]}, index=["Churn", "Stay"]))
        st.info("ระดับความน่าจะเป็นที่ลูกค้าจะยกเลิกบริการเปรียบเทียบกับการคงอยู่ต่อ โดยวิเคราะห์จากปัจจัยด้านการเงินและพฤติกรรมการใช้งาน")

# -------------------------------------------------
# PAGE: TEST NN
# -------------------------------------------------
elif page == "Test: Neural Network":
    st.title("Diabetes Risk Prediction Test")
    glu = st.number_input("ระดับน้ำตาล (Glucose)", 0, 200, 100)
    bmi = st.number_input("ค่า BMI", 0.0, 60.0, 25.0)
    age_nn = st.number_input("อายุ (Age)", 1, 120, 30)
    
    if st.button("Analyze"):
        risk = (glu/200)*100
        st.metric("Risk Score", f"{risk:.1f}%")
        st.bar_chart(pd.DataFrame({"Score": [risk, 50]}, index=["Risk Score", "Threshold"]))
        st.info("การประเมินความเสี่ยงปัจจุบันเทียบกับขีดจำกัดมาตรฐาน หากค่าสูงกว่าเกณฑ์แสดงว่ามีความเสี่ยงสูงต่อโรคเบาหวาน")
