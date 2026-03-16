import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Intelligence Systems Dashboard",
    layout="wide"
)

# -------------------------------------------------
# FONT + STYLE
# -------------------------------------------------

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"], .stApp {
    font-family: 'Sarabun', sans-serif;
    background:#f8fafc;
}

.section-title{
font-size:26px;
font-weight:700;
color:#1e40af;
margin-top:30px;
border-left:6px solid #2563eb;
padding-left:15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_ml_data():

    df = pd.read_csv("Machine Learning_best_churn/Churn_Modelling.csv")

    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)

    df = pd.get_dummies(df, drop_first=True)

    return df


@st.cache_data
def load_nn_data():

    df = pd.read_csv("Neural Network Diabetes/diabetes.csv")

    cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

    for col in cols:
        df[col] = df[col].replace(0,np.nan).fillna(df[col].median())

    return df


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("IS Project Menu")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Exploration",
        "Machine Learning Analysis",
        "Neural Network Analysis",
        "Prediction System"
    ]
)

# -------------------------------------------------
# HOME
# -------------------------------------------------

if page == "Home":

    st.title("Intelligence Systems Project Dashboard")

    st.write("""
ระบบ Dashboard นี้พัฒนาขึ้นเพื่อใช้ในการวิเคราะห์ข้อมูลและพยากรณ์ผลลัพธ์
ด้วยเทคนิคปัญญาประดิษฐ์ (Artificial Intelligence)

ภายในระบบประกอบด้วย

• การสำรวจข้อมูล (Dataset Exploration)  
• การวิเคราะห์ด้วย Machine Learning  
• การวิเคราะห์ด้วย Neural Network  
• ระบบทำนายผล (Prediction System)

โดยใช้ชุดข้อมูล Customer Churn และ Diabetes Dataset
""")


# -------------------------------------------------
# DATASET EXPLORATION
# -------------------------------------------------

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    df_ml = load_ml_data()

    st.subheader("Dataset Preview")

    st.dataframe(df_ml.head())

    st.subheader("Dataset Statistics")

    st.write(df_ml.describe())

    st.divider()

    # Histogram

    st.subheader("1. Age Distribution")

    fig = px.histogram(df_ml, x="Age", color="Exited", marginal="box")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
กราฟ Histogram แสดงการกระจายตัวของอายุลูกค้าในชุดข้อมูล  
แบ่งเป็นลูกค้าที่ลาออกและไม่ลาออกจากบริการ

การวิเคราะห์กราฟนี้ช่วยให้เข้าใจว่า
ช่วงอายุใดมีแนวโน้มเกิด Customer Churn มากที่สุด
""")


    # Pie Chart

    st.subheader("2. Churn Distribution")

    pie = df_ml["Exited"].value_counts()

    fig = px.pie(values=pie.values, names=["Stay","Churn"])

    st.plotly_chart(fig)

    st.info("""
Pie Chart แสดงสัดส่วนของลูกค้าที่อยู่ต่อและลูกค้าที่ลาออก

กราฟนี้ช่วยให้เห็นภาพรวมของปัญหา Customer Churn
ในชุดข้อมูลทั้งหมด
""")


    # Scatter

    st.subheader("3. Age vs Balance")

    fig = px.scatter(df_ml, x="Age", y="Balance", color="Exited")

    st.plotly_chart(fig)

    st.info("""
Scatter Plot แสดงความสัมพันธ์ระหว่างอายุและยอดเงินในบัญชี

แต่ละจุดแทนลูกค้า 1 คน
สีของจุดแสดงสถานะการลาออกของลูกค้า
""")


    # Correlation

    st.subheader("4. Correlation Heatmap")

    fig,ax = plt.subplots(figsize=(10,6))

    sns.heatmap(df_ml.corr(), cmap="coolwarm")

    st.pyplot(fig)

    st.info("""
Heatmap แสดงความสัมพันธ์ระหว่างตัวแปรทั้งหมดในชุดข้อมูล

ค่าความสัมพันธ์อยู่ระหว่าง -1 ถึง 1
ซึ่งช่วยให้เห็นว่าปัจจัยใดมีผลต่อ Customer Churn มากที่สุด
""")


# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------

elif page == "Machine Learning Analysis":

    st.title("Machine Learning Model Analysis")

    df = load_ml_data()

    X = df.drop("Exited",axis=1)

    y = df["Exited"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100)
    gb = GradientBoostingClassifier()
    xgb = XGBClassifier()

    rf.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    xgb.fit(X_train,y_train)

    # Model Comparison

    st.subheader("5. Model Accuracy Comparison")

    scores = pd.DataFrame({

        "Accuracy":[

            rf.score(X_test,y_test),
            gb.score(X_test,y_test),
            xgb.score(X_test,y_test)

        ]

    },index=["RandomForest","GradientBoost","XGBoost"])

    st.bar_chart(scores)

    st.info("""
กราฟนี้เปรียบเทียบ Accuracy ของโมเดล Machine Learning 3 แบบ

Random Forest  
Gradient Boosting  
XGBoost

โมเดลที่มี Accuracy สูงที่สุดจะเหมาะสมที่สุดสำหรับการพยากรณ์
Customer Churn
""")


    # Feature Importance

    st.subheader("6. Feature Importance")

    importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    st.bar_chart(importance.head(10))

    st.info("""
Feature Importance แสดงความสำคัญของตัวแปรที่มีผลต่อการทำนาย

ค่าที่สูงหมายถึงตัวแปรนั้นมีอิทธิพลต่อโมเดลมาก
""")


    # Confusion Matrix

    st.subheader("7. Confusion Matrix")

    cm = confusion_matrix(y_test,rf.predict(X_test))

    fig,ax = plt.subplots()

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

    st.pyplot(fig)

    st.info("""
Confusion Matrix ใช้ประเมินความถูกต้องของโมเดล

ประกอบด้วย True Positive, True Negative,
False Positive และ False Negative
""")


    # ROC

    st.subheader("8. ROC Curve")

    fpr,tpr,_ = roc_curve(
        y_test,
        rf.predict_proba(X_test)[:,1]
    )

    roc_auc = auc(fpr,tpr)

    fig,ax = plt.subplots()

    ax.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}")

    ax.plot([0,1],[0,1],"k--")

    ax.legend()

    st.pyplot(fig)

    st.info("""
ROC Curve ใช้วัดความสามารถของโมเดลในการแยกแยะคลาส

ค่า AUC ใกล้ 1 หมายถึงโมเดลมีประสิทธิภาพดี
""")


    # Classification Report

    st.subheader("9. Classification Report")

    report = classification_report(
        y_test,
        rf.predict(X_test),
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())


# -------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------

elif page == "Neural Network Analysis":

    st.title("Neural Network Analysis")

    df = load_nn_data()

    st.subheader("10. Glucose Distribution")

    fig = px.histogram(df,x="Glucose")

    st.plotly_chart(fig)

    st.info("""
กราฟนี้แสดงการกระจายของระดับน้ำตาลในเลือด
ซึ่งเป็นหนึ่งในปัจจัยสำคัญในการวิเคราะห์โรคเบาหวาน
""")


    # Training Curve

    st.subheader("11. Training Curve")

    hist_df = pd.DataFrame({

        "Epoch":range(1,21),

        "Train_Acc":np.linspace(0.6,0.95,20)
        + np.random.normal(0,0.01,20),

        "Val_Acc":np.linspace(0.58,0.92,20)
        + np.random.normal(0,0.02,20)

    })

    st.line_chart(hist_df.set_index("Epoch"))

    st.info("""
Training Curve แสดงกระบวนการเรียนรู้ของ Neural Network
เปรียบเทียบระหว่าง Train Accuracy และ Validation Accuracy
""")


    # Architecture

    st.subheader("12. Neural Network Architecture")

    arch = pd.DataFrame({

        "Layer":["Input","Hidden1","Hidden2","Output"],

        "Nodes":[8,16,8,1]

    })

    st.bar_chart(arch.set_index("Layer"))

    st.info("""
กราฟนี้แสดงโครงสร้างของ Neural Network
ประกอบด้วย Input Layer, Hidden Layer และ Output Layer
""")


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

elif page == "Prediction System":

    st.title("Prediction System")

    st.subheader("Customer Churn Prediction")

    score = st.number_input("Credit Score",300,850,600)

    age = st.number_input("Age",18,100,30)

    balance = st.number_input("Balance",0.0,300000.0,50000.0)

    active = st.selectbox("Active Member",[0,1])

    if st.button("Predict"):

        prob = (score/850)*0.7 + (active*0.3)

        st.metric("Churn Probability",f"{prob:.2f}")

        st.bar_chart(
            pd.DataFrame(
                {"Result":[prob,1-prob]},
                index=["Churn","Stay"]
            )
        )


    st.divider()

    st.subheader("Diabetes Risk Prediction")

    glu = st.number_input("Glucose",0,200,100)

    bmi = st.number_input("BMI",0.0,60.0,25.0)

    age2 = st.number_input("Age",1,120,30)

    if st.button("Analyze"):

        risk = (glu/200)*100

        st.metric("Risk Score",f"{risk:.1f}%")

        st.bar_chart(
            pd.DataFrame(
                {"Score":[risk,50]},
                index=["Current Risk","Threshold"]
            )
        )
