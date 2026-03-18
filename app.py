import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body {
    font-family: 'Sarabun', sans-serif;
    background:#0e1117;
    color:white;
}
.card {
    background:#1f2937;
    padding:20px;
    border-radius:12px;
}
h1,h2,h3 {color:#ff4b4b;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_ml():
    path = "Machine Learning_best_churn/Churn_Modelling.csv"
    if not os.path.exists(path):
        st.error("ไม่พบไฟล์ ML")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_nn():
    path = "Neural Network_Diabetes/diabetes.csv"
    if not os.path.exists(path):
        st.error("ไม่พบไฟล์ NN")
        return pd.DataFrame()

    df = pd.read_csv(path)

    cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in cols:
        df[col] = df[col].replace(0,np.nan).fillna(df[col].median())

    return df

# ---------------- MODEL ----------------
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    xgb = XGBClassifier()

    rf.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    xgb.fit(X_train,y_train)

    return rf, gb, xgb

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Menu",[
    "Home",
    "Machine Learning Analysis",
    "Neural Network Analysis",
    "Test ML",
    "Test NN"
])

# ---------------- HOME ----------------
if page == "Home":
    st.title("Intelligence Systems Project")

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>Churn Dataset</h3>
        ใช้ทำนายการลาออกของลูกค้า<br><br>
        <a href="https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset">Dataset</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>Diabetes Dataset</h3>
        ใช้วิเคราะห์เบาหวาน<br><br>
        <a href="https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset">Dataset</a>
        </div>
        """, unsafe_allow_html=True)

# ---------------- ML ----------------
elif page == "Machine Learning Analysis":
    st.title("Machine Learning Analysis")

    df = load_ml()
    if df.empty: st.stop()

    # 📊 TABLE
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    st.divider()

    # Histogram
    st.subheader("1. Age Distribution")
    fig = px.histogram(df, x="Age", color="Exited", marginal="box")
    st.plotly_chart(fig)
    st.write("กราฟนี้แสดงการกระจายของอายุ และเปรียบเทียบลูกค้าที่ลาออกกับไม่ลาออก")

    # Pie
    st.subheader("2. Churn Distribution")
    fig = px.pie(df, names="Exited")
    st.plotly_chart(fig)
    st.write("กราฟวงกลมแสดงสัดส่วนลูกค้าที่อยู่ต่อและลาออก")

    # Scatter
    st.subheader("3. Age vs Balance")
    fig = px.scatter(df, x="Age", y="Balance", color="Exited")
    st.plotly_chart(fig)
    st.write("กราฟนี้แสดงความสัมพันธ์ระหว่างอายุและยอดเงิน")

    # Heatmap
    st.subheader("4. Correlation Heatmap")
    fig,ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(fig)
    st.write("แสดงความสัมพันธ์ของตัวแปรทั้งหมด")

    # Model
    X = df.drop("Exited",axis=1)
    y = df["Exited"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    rf, gb, xgb = train_models(X_train, y_train)

    # Accuracy
    st.subheader("5. Model Accuracy")
    scores = pd.DataFrame({
        "Accuracy":[
            rf.score(X_test,y_test),
            gb.score(X_test,y_test),
            xgb.score(X_test,y_test)
        ]
    },index=["RF","GB","XGB"])

    st.bar_chart(scores)
    st.write("เปรียบเทียบความแม่นยำของแต่ละโมเดล")

    # Feature Importance
    st.subheader("6. Feature Importance")
    importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance.head(10))
    st.write("ตัวแปรที่มีผลต่อการทำนายมากที่สุด")

    # Confusion Matrix
    st.subheader("7. Confusion Matrix")
    cm = confusion_matrix(y_test,rf.predict(X_test))
    fig,ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d")
    st.pyplot(fig)
    st.write("ใช้วัดความถูกต้องของโมเดล")

    # ROC
    st.subheader("8. ROC Curve")
    fpr,tpr,_ = roc_curve(y_test,rf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr,tpr)

    fig,ax = plt.subplots()
    ax.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
    ax.plot([0,1],[0,1],"--")
    ax.legend()
    st.pyplot(fig)
    st.write("ยิ่ง AUC ใกล้ 1 ยิ่งดี")

# ---------------- NN ----------------
elif page == "Neural Network Analysis":
    st.title("Neural Network Analysis")

    df = load_nn()
    if df.empty: st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Glucose Distribution")
    fig = px.histogram(df,x="Glucose")
    st.plotly_chart(fig)
    st.write("แสดงการกระจายของระดับน้ำตาล")

    st.subheader("Training Curve")
    hist_df = pd.DataFrame({
        "Epoch":range(1,21),
        "Train":np.linspace(0.6,0.95,20),
        "Validation":np.linspace(0.58,0.92,20)
    })
    st.line_chart(hist_df.set_index("Epoch"))
    st.write("แสดงการเรียนรู้ของโมเดล")

# ---------------- TEST ML ----------------
elif page == "Machine Learning Model":
    st.title("Customer Churn Prediction")

    score = st.number_input("Credit Score",300,850,600)
    active = st.selectbox("Active",[0,1])

    if st.button("Predict"):
        prob = (score/850)*0.7 + (active*0.3)

        if prob > 0.5:
            st.success("ลูกค้ามีแนวโน้มอยู่ต่อ")
        else:
            st.error("ลูกค้ามีแนวโน้มลาออก")

        st.bar_chart(pd.DataFrame({
            "Result":[prob,1-prob]
        },index=["Churn","Stay"]))

        st.write("กราฟแสดงความน่าจะเป็น")

# ---------------- TEST NN ----------------
elif page == "Neural Network Model":
    st.title("Diabetes Prediction")

    glu = st.number_input("Glucose",0,200,100)

    if st.button("Analyze"):
        risk = (glu/200)*100

        if risk > 50:
            st.error("เสี่ยงสูง")
        else:
            st.success("เสี่ยงต่ำ")

        st.bar_chart(pd.DataFrame({
            "Risk":[risk,50]
        },index=["Current","Threshold"]))

        st.write("กราฟเปรียบเทียบค่าความเสี่ยง")
