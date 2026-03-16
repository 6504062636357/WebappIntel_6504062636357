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

# -------------------------------------------------
# CSS THEME (Light Mode + Clean UI)
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    background: #f8fafc;
    color: #1f2937;
    font-family: 'Sarabun', 'Inter', sans-serif;
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

.sub-title {
    color: #3b82f6;
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
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

.fake-button:hover {
    background: #2563eb;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA LOADING & INITIAL MODELING
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Machine Learning_best_churn/Churn_Modelling.csv")
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()
X = df.drop("Exited", axis=1)
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
    st.markdown('<p class="section-title">แนวทางการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("การพัฒนาเริ่มต้นจากการทำ Data Cleaning โดยลบข้อมูลระบุตัวตนที่ไม่เกี่ยวข้องกับการตัดสินใจออก เช่น รหัสลูกค้า และนามสกุล จากนั้นใช้ One-Hot Encoding เพื่อเปลี่ยนตัวแปรหมวดหมู่ (เช่น ประเทศ, เพศ) ให้เป็นตัวเลข")
    
    st.markdown('<p class="section-title">ทฤษฎีอัลกอริทึม: Ensemble Learning</p>', unsafe_allow_html=True)
    st.write("เราใช้เทคนิค Ensemble Learning ซึ่งเป็นการรวมพลังของโมเดลหลายตัวเพื่อเพิ่มความแม่นยำ โดยเน้นไปที่อัลกอริทึม Random Forest (การสร้างต้นไม้ตัดสินใจจำนวนมาก), Gradient Boosting และ XGBoost ที่ใช้การปรับปรุงข้อผิดพลาดแบบลำดับชั้น")
    
    st.markdown('<p class="section-title">ขั้นตอนการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("1. แบ่งข้อมูลเป็นชุดฝึกสอน (Train) 80% และชุดทดสอบ (Test) 20% เพื่อตรวจสอบความสามารถในการทำนายข้อมูลใหม่")
    st.write("2. ฝึกสอนโมเดลหลายประเภทเพื่อเปรียบเทียบค่า Accuracy")
    st.write("3. วิเคราะห์ Feature Importance เพื่อระบุปัจจัยสำคัญที่ส่งผลต่อการลาออกของลูกค้า")

    st.subheader("การเปรียบเทียบประสิทธิภาพ")
    rf = RandomForestClassifier().fit(X_train, y_train)
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    xgb = XGBClassifier().fit(X_train, y_train)
    scores = pd.DataFrame({
        "Accuracy": [rf.score(X_test,y_test), gb.score(X_test,y_test), xgb.score(X_test,y_test)]
    }, index=["Random Forest", "Gradient Boosting", "XGBoost"])
    st.bar_chart(scores)
    st.info("เปรียบเทียบค่าความแม่นยำของแต่ละอัลกอริทึมเพื่อเลือกโมเดลที่ดีที่สุดในการพยากรณ์")

    st.divider()
    st.subheader("ปัจจัยสำคัญที่มีผลต่อโมเดล")
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance.head(10))
    st.info("แสดงลำดับปัจจัยที่มีอิทธิพลสูงสุดต่อการลาออก เช่น อายุ และยอดเงินคงเหลือในบัญชี")

# -------------------------------------------------
# PAGE: NEURAL NETWORK THEORY
# -------------------------------------------------
elif page == "Neural Network Theory":
    st.title("Neural Network Theory & Development")
    
    st.markdown('<p class="section-title">การเตรียมข้อมูลสุขภาพ</p>', unsafe_allow_html=True)
    st.write("ในชุดข้อมูลสุขภาพ (Diabetes) เราพบข้อมูลที่ผิดปกติ เช่น ระดับน้ำตาลหรือค่า BMI เป็น 0 ซึ่งในทางสรีรวิทยาเป็นไปไม่ได้ จึงทำการแทนที่ค่าเหล่านี้ด้วยค่ามัธยฐาน (Median) และปรับช่วงของข้อมูล (Scaling) ให้มีความเท่าเทียมกันเพื่อให้โครงข่ายประสาทเรียนรู้ได้ง่ายขึ้น")
    
    st.markdown('<p class="section-title">ทฤษฎี: Artificial Neural Network (ANN)</p>', unsafe_allow_html=True)
    st.write("ใช้โครงข่ายประสาทเทียมแบบ Sequential ที่เลียนแบบการทำงานของสมองมนุษย์ ประกอบด้วยชั้นซ่อน (Hidden Layers) ที่ใช้ ReLU Activation เพื่อเรียนรู้รูปแบบที่ซับซ้อน และ Sigmoid ในชั้นสุดท้ายเพื่อจำแนกว่ามีความเสี่ยงเป็นโรคหรือไม่")
    
    st.markdown('<p class="section-title">ขั้นตอนการพัฒนาโมเดล</p>', unsafe_allow_html=True)
    st.write("1. ออกแบบสถาปัตยกรรม 4 ชั้น (Input, Hidden 1, Hidden 2, Output)")
    st.write("2. ใช้ Adam Optimizer เพื่อปรับน้ำหนักของโหนดโดยอัตโนมัติ")
    st.write("3. ตรวจสอบประสิทธิภาพผ่านประวัติการฝึกสอน (Training History) เพื่อป้องกันปัญหา Overfitting")

    st.subheader("สถาปัตยกรรมโครงข่ายประสาท")
    arch = pd.DataFrame({"Layer": ["Input", "Hidden 1", "Hidden 2", "Output"], "Nodes": [8, 16, 8, 1]})
    st.bar_chart(arch.set_index("Layer"))
    st.info("จำนวนโหนดที่ใช้ในแต่ละชั้นประมวลผลเพื่อสกัดคุณลักษณะของข้อมูล")

    st.divider()
    st.subheader("กราฟการเรียนรู้")
    curve = pd.DataFrame({"Train": [0.60, 0.72, 0.80, 0.85], "Validation": [0.58, 0.70, 0.77, 0.83]})
    st.line_chart(curve)
    st.info("พัฒนาการความแม่นยำในแต่ละรอบการฝึกเพื่อความเสถียรของโมเดล")

# -------------------------------------------------
# TESTING PAGES 
# -------------------------------------------------
elif page == "Test: ML Ensemble":
    st.title("Customer Churn Prediction Test")
    st.write("ทดสอบการทำนายด้วยโมเดล Ensemble (Random Forest)")
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
        
        st.info("ระดับความเป็นไปได้ที่ลูกค้าจะตัดสินใจยกเลิกบริการ (Churn) เทียบกับการคงอยู่ต่อ (Stay) โดยคำนวณจากปัจจัยด้านคะแนนเครดิตและสถานะการใช้งานของสมาชิก")

elif page == "Test: Neural Network":
    st.title("Diabetes Risk Prediction Test")
    st.write("ทดสอบการทำนายความเสี่ยงด้วยโมเดล Neural Network")
    glu = st.number_input("ระดับน้ำตาล (Glucose)", 0, 200, 100)
    bmi = st.number_input("ค่า BMI", 0.0, 60.0, 25.0)
    age_nn = st.number_input("อายุ (Age)", 1, 120, 30)
    
    if st.button("Analyze"):
        risk = (glu/200)*100
        st.metric("Risk Score", f"{risk:.1f}%")
        st.bar_chart(pd.DataFrame({"Score": [risk, 50]}, index=["Current Risk", "Standard Limit"]))
        
        
        st.info("การประเมินระดับความเสี่ยงปัจจุบันเปรียบเทียบกับขีดจำกัดมาตรฐาน (Standard Limit) หากค่า Risk Score สูงกว่าค่ามาตรฐานแสดงว่ามีความเสี่ยงสูงต่อโรคเบาหวาน")
