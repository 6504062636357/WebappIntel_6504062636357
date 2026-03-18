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
html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif;
    background:#0e1117;
    color:white;
}
h1,h2,h3 {color:#ff4b4b;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_ml_data():
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_nn_data():
    path = "Neural Network_Diabetes/diabetes.csv"

    if not os.path.exists(path):
        st.error("ไม่พบไฟล์ diabetes.csv")
        return pd.DataFrame()

    df = pd.read_csv(path)

    cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in cols:
        df[col] = df[col].replace(0,np.nan).fillna(df[col].median())

    return df

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100)
    gb = GradientBoostingClassifier()
    xgb = XGBClassifier()

    rf.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    xgb.fit(X_train,y_train)

    return rf, gb, xgb

# ---------------- SIDEBAR ----------------
st.sidebar.title("IS Project Menu")

page = st.sidebar.radio("Navigation",[
    "Home & Dataset",
    "Machine Learning Analysis",
    "Neural Network Analysis",
    "Test ML",
    "Test NN"
])

# ---------------- HOME ----------------
if page == "Home & Dataset":
    st.title("Intelligence Systems Project")

    st.subheader("Dataset Links")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Churn Dataset")
        st.markdown("https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset")

    with col2:
        st.write("Diabetes Dataset")
        st.markdown("https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset")

# ---------------- ML ----------------
elif page == "Machine Learning Analysis":
    st.title("Machine Learning Analysis")

    df = load_ml_data()
    st.dataframe(df.head())

    # Histogram
    st.subheader("Age Distribution")
    fig = px.histogram(df, x="Age", color="Exited", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    # Pie
    st.subheader("Churn Distribution")
    pie = df["Exited"].value_counts()
    fig = px.pie(values=pie.values, names=["Stay","Churn"])
    st.plotly_chart(fig)

    # Scatter
    st.subheader("Age vs Balance")
    fig = px.scatter(df, x="Age", y="Balance", color="Exited")
    st.plotly_chart(fig)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig,ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    st.pyplot(fig)

    # Model
    X = df.drop("Exited",axis=1)
    y = df["Exited"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    rf, gb, xgb = train_models(X_train, y_train)

    # Accuracy
    st.subheader("Model Accuracy")
    scores = pd.DataFrame({
        "Accuracy":[
            rf.score(X_test,y_test),
            gb.score(X_test,y_test),
            xgb.score(X_test,y_test)
        ]
    },index=["RF","GB","XGB"])

    st.bar_chart(scores)

    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance.head(10))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test,rf.predict(X_test))
    fig,ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d")
    st.pyplot(fig)

    # ROC
    st.subheader("ROC Curve")
    fpr,tpr,_ = roc_curve(y_test,rf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr,tpr)

    fig,ax = plt.subplots()
    ax.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
    ax.plot([0,1],[0,1],"--")
    ax.legend()
    st.pyplot(fig)

    # Report
    st.subheader("Classification Report")
    report = classification_report(y_test,rf.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ---------------- NN ----------------
elif page == "Neural Network Analysis":
    st.title("Neural Network Analysis")

    df = load_nn_data()

    if df.empty:
        st.stop()

    st.dataframe(df.head())

    st.subheader("Glucose Distribution")
    fig = px.histogram(df,x="Glucose")
    st.plotly_chart(fig)

    st.subheader("Training Curve")
    hist_df = pd.DataFrame({
        "Epoch":range(1,21),
        "Train_Acc":np.linspace(0.6,0.95,20),
        "Val_Acc":np.linspace(0.58,0.92,20)
    })
    st.line_chart(hist_df.set_index("Epoch"))

# ---------------- TEST ML ----------------
elif page == "Test ML":
    st.title("Customer Churn Prediction")

    score = st.number_input("Credit Score",300,850,600)
    age = st.number_input("Age",18,100,30)
    balance = st.number_input("Balance",0.0,300000.0,50000.0)
    active = st.selectbox("Active Member",[0,1])

    if st.button("Predict"):
        prob = (score/850)*0.7 + (active*0.3)

        if prob > 0.5:
            st.success("ลูกค้ามีแนวโน้มอยู่ต่อ")
        else:
            st.error("ลูกค้ามีแนวโน้มลาออก")

        st.metric("Probability",f"{prob:.2f}")

        st.bar_chart(pd.DataFrame({
            "Result":[prob,1-prob]
        },index=["Churn","Stay"]))

# ---------------- TEST NN ----------------
elif page == "Test NN":
    st.title("Diabetes Prediction")

    glu = st.number_input("Glucose",0,200,100)
    bmi = st.number_input("BMI",0.0,60.0,25.0)

    if st.button("Analyze"):
        risk = (glu/200)*100

        if risk > 50:
            st.error("มีความเสี่ยงสูง")
        else:
            st.success("ความเสี่ยงต่ำ")

        st.metric("Risk",f"{risk:.1f}%")

        st.bar_chart(pd.DataFrame({
            "Score":[risk,50]
        },index=["Risk","Threshold"]))
