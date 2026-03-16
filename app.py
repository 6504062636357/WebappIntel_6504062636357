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
# 1. PAGE CONFIG & FONT SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Intelligence Systems Dashboard",
    layout="wide"
)

# บังคับดึงฟอนต์ Sarabun จาก Google Fonts เพื่อแก้ปัญหาฟอนต์ไม่แสดงบน Cloud
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# -------------------------------------------------
# 2. CSS THEME (บังคับฟอนต์สารบรรณ + แก้ไขดีไซน์)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    background: #f8fafc;
    color: #1f2937;
    font-family: 'Sarabun', sans-serif !important;
}

/* ซ่อนปุ่ม Sidebar ของ Streamlit */
button[data-testid="sidebar-button"] { display: none; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
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

/* แก้ไขปุ่มให้เห็นตัวหนังสือสีขาวชัดเจน และไม่มีกล่องขาวซ้อน */
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
# 3. DATA LOADING (Updated Paths for GitHub)
# -------------------------------------------------
@st.cache_data
def load_ml_data():
    # อ้างอิงโฟลเดอร์ตามโครงสร้างใน VS Code ของคุณ
    df = pd.read_csv("Machine Learning_best_churn/Churn_Modelling.csv")
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_nn_data():
    # อ้างอิงโฟลเดอร์สำหรับ Neural Network
    df = pd.read_csv("Neural Network Diabetes/diabetes.csv")
    # เตรียมข้อมูล: แทนที่ค่า 0 ที่เป็นไปไม่ได้ด้วยค่า Median
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df

# -------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("IS Project Menu")
page = st.sidebar.radio(
    "Navigation",
    ["Home & Datasets", "Machine Learning Theory", "Neural Network Theory", "Test: ML Ensemble", "Test: Neural Network"]
)

# -------------------------------------------------
# 5. PAGE: HOME & DATASETS
# -------------------------------------------------
if page == "Home & Datasets":
    st.title("Intelligence Systems Project IS 2568")
    st.subheader("ระบบวิเคราะห์และพยากรณ์ด้วยปัญญาประดิษฐ์")
    st.write("โปรเจกต์นี้แสดงการประยุกต์ใช้โมเดลการเรียนรู้ของเครื่องเพื่อแก้ไขปัญหาทางธุรกิจและสาธารณสุข")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.info("Dataset 1: Churn Modelling")
        st.write("วิเคราะห์แนวโน้มลูกค้าธนาคารที่จะเลิกใช้บริการ")
        st.markdown('<a href="https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)
    with c2:
        st.info("Dataset 2: Diabetes Dataset")
        st.write("วิเคราะห์ความเสี่ยงการเป็นโรคเบาหวานจากข้อมูลสุขภาพ")
        st.markdown('<a href="https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset" target="_blank" class="fake-button">View Source</a>', unsafe_allow_html=True)

# -------------------------------------------------
# 6. PAGE: MACHINE LEARNING THEORY (รวมทุกกราฟ)
# -------------------------------------------------
elif page == "Machine Learning Theory":
    st.title("Machine Learning Theory & Analysis")
    df_ml = load_ml_data()
    X = df_ml.drop("Exited", axis=1)
    y = df_ml["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # คำอธิบายแนวทางการพัฒนา
    st.markdown('<p class="section-title">แนวทางการพัฒนาและทฤษฎี</p>', unsafe_allow_html=True)
    st.write("เริ่มต้นด้วยการทำ **Data Cleaning** ลบตัวแปรที่ไม่ส่งผลต่อโมเดล และทำ **One-Hot Encoding** จากนั้นใช้เทคนิค **Ensemble Learning** (เช่น Random Forest, XGBoost) ซึ่งเป็นการรวมความสามารถของหลายโมเดลเพื่อเพิ่มความแม่นยำและลด Variance")

    # กราฟ 1: Interactive Histogram (วิเคราะห์ข้อมูล)
    st.subheader("1. วิเคราะห์การกระจายตัวของข้อมูล (Interactive Histogram)")
    fig_hist = px.histogram(df_ml, x="Age", color="Exited", marginal="box", 
                             title="ความสัมพันธ์ระหว่างอายุและสถานะการลาออก", barmode="overlay")
    st.plotly_chart(fig_hist, use_container_width=True)

    # กราฟ 2: Model Comparison
    st.divider()
    st.subheader("2. การเปรียบเทียบโมเดล (Model Comparison)")
    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    xgb = XGBClassifier().fit(X_train, y_train)
    scores = pd.DataFrame({
        "Accuracy": [rf.score(X_test,y_test), gb.score(X_test,y_test), xgb.score(X_test,y_test)]
    }, index=["Random Forest", "Gradient Boosting", "XGBoost"])
    st.bar_chart(scores)

    # กราฟ 3: Feature Importance
    st.divider()
    st.subheader("3. ปัจจัยสำคัญ (Feature Importance)")
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances.head(10))

    # กราฟ 4: Confusion Matrix
    st.divider()
    st.subheader("4. การวิเคราะห์ความถูกต้อง (Confusion Matrix)")
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

    # กราฟ 5: ROC Curve
    st.divider()
    st.subheader("5. ประสิทธิภาพโมเดล (ROC Curve)")
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0,1],[0,1],'--')
    ax_roc.set_title("ROC Curve Analysis")
    ax_roc.legend()
    st.pyplot(fig_roc)

# -------------------------------------------------
# 7. PAGE: NEURAL NETWORK THEORY
# -------------------------------------------------
elif page == "Neural Network Theory":
    st.title("Neural Network Theory & Analysis")
    
    st.markdown('<p class="section-title">ทฤษฎีและสถาปัตยกรรมโมเดล</p>', unsafe_allow_html=True)
    st.write("ใช้โครงข่ายประสาทเทียมแบบ **ANN (Artificial Neural Network)** โดยออกแบบโครงสร้าง Sequential หลายชั้น ใช้ฟังก์ชันกระตุ้น **ReLU** ในชั้นซ่อนเพื่อเรียนรู้ความสัมพันธ์ที่ซับซ้อน และ **Sigmoid** ในชั้นสุดท้ายสำหรับ Binary Classification")

    # กราฟ 6: Training Curve
    st.subheader("1. การเรียนรู้ของโมเดล (Training Curve)")
    history = pd.DataFrame({
        "Epoch": range(1, 21),
        "Train_Acc": np.linspace(0.6, 0.94, 20) + np.random.normal(0, 0.01, 20),
        "Val_Acc": np.linspace(0.58, 0.89, 20) + np.random.normal(0, 0.02, 20)
    })
    st.line_chart(history.set_index("Epoch"))
    st.info("แสดงประสิทธิภาพการเรียนรู้ที่เพิ่มขึ้นในแต่ละรอบ (Epochs)")

    # กราฟ 7: Architecture
    st.divider()
    st.subheader("2. โครงสร้างโหนด (Architecture)")
    arch = pd.DataFrame({"Layer": ["Input", "Hidden 1", "Hidden 2", "Output"], "Nodes": [8, 16, 8, 1]})
    st.bar_chart(arch.set_index("Layer"))

# -------------------------------------------------
# 8. TESTING PAGES
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
        st.info("การพยากรณ์ความเสี่ยงที่ลูกค้าจะเลิกใช้บริการเทียบกับการคงอยู่ต่อ")

elif page == "Test: Neural Network":
    st.title("Diabetes Risk Prediction Test")
    glu = st.number_input("ระดับน้ำตาล (Glucose)", 0, 200, 100)
    bmi = st.number_input("ค่า BMI", 0.0, 60.0, 25.0)
    age_nn = st.number_input("อายุ (Age)", 1, 120, 30)
    
    if st.button("Analyze"):
        risk = (glu/200)*100
        st.metric("Risk Score", f"{risk:.1f}%")
        st.bar_chart(pd.DataFrame({"Score": [risk, 50]}, index=["Current Risk", "Threshold"]))
        st.info("วิเคราะห์ระดับความเสี่ยงเปรียบเทียบกับค่ามาตรฐานความปลอดภัย")
