import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

# ── load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("Telco_customer_churn.xlsx")
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(0)
    drop_columns = ['CustomerID','Count','Country','State','Lat Long',
                    'Longitude','Latitude','Zip Code','Churn Label',
                    'Churn Score','CLTV','Churn Reason','City']
    df = df.drop(columns=drop_columns)
    return df

# ── train models ──────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df_encoded = pd.get_dummies(_df, drop_first=True)
    X = df_encoded.drop(columns=['Churn Value'])
    Y = df_encoded['Churn Value']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    rf = RandomForestClassifier(n_estimators=300,max_depth=10,
                                class_weight='balanced',random_state=42)
    rf.fit(X_train,Y_train)

    xgb = XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
                        scale_pos_weight=3,use_label_encoder=False,
                        eval_metric='logloss',random_state=42,verbosity=0)
    xgb.fit(X_train,Y_train)

    # segmentation
    y_prob = rf.predict_proba(X)[:,1]
    seg = pd.DataFrame({
        'Tenure Months'    : X['Tenure Months'],
        'Monthly Charges'  : X['Monthly Charges'],
        'Total Charges'    : X['Total Charges'],
        'Churn Probability': y_prob
    })
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(seg)
    kmeans  = KMeans(n_clusters=3,random_state=42,n_init=10)
    seg['Cluster'] = kmeans.fit_predict(scaled)
    cluster_names  = {0:'Budget Loyal Customers',
                      1:'High Risk New Customers',
                      2:'Loyal Premium Customers'}
    seg['Cluster Segment'] = seg['Cluster'].map(cluster_names)

    return rf, xgb, X, Y, X_test, Y_test, seg

# ── load ──────────────────────────────────────────────────────
df = load_data()
rf, xgb, X, Y, X_test, Y_test, seg_data = train_models(df)

# ── sidebar ───────────────────────────────────────────────────
st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "🏠 Overview",
    "📊 Model Comparison" ,
    "🔮 Predict Customer",
    "🗂️ Segmentation"
])

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 Customer Churn Dashboard")
    st.markdown("---")

    total    = len(df)
    churned  = int(Y.sum())
    retained = total - churned
    rate     = round(churned/total*100,2)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned",         f"{churned:,}")
    c3.metric("Retained",        f"{retained:,}")
    c4.metric("Churn Rate",      f"{rate}%")

    st.markdown("---")
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract Type")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='Contract',hue='Churn Value',data=df,ax=ax)
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn by Internet Service")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='Internet Service',hue='Churn Value',data=df,ax=ax)
        ax.set_xlabel('Internet Service')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Monthly Charges Distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['Monthly Charges'],bins=30,kde=True,ax=ax)
        ax.set_xlabel('Monthly Charges')
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Tenure Months Distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['Tenure Months'],bins=30,kde=True,ax=ax)
        ax.set_xlabel('Tenure Months')
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Accuracy Comparison")
    st.markdown("---")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Predictions
    rf_pred  = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # Probabilities
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [
            accuracy_score(Y_test, rf_pred),
            accuracy_score(Y_test, xgb_pred)
        ],
        "Precision": [
            precision_score(Y_test, rf_pred),
            precision_score(Y_test, xgb_pred)
        ],
        "Recall": [
            recall_score(Y_test, rf_pred),
            recall_score(Y_test, xgb_pred)
        ],
        "F1 Score": [
            f1_score(Y_test, rf_pred),
            f1_score(Y_test, xgb_pred)
        ]
    })

    st.subheader("📌 Performance Table")
    st.dataframe(metrics.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }), use_container_width=True)

    st.markdown("---")

    # Confusion Matrix
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Confusion Matrix")
        cm_rf = confusion_matrix(Y_test, rf_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("XGBoost Confusion Matrix")
        cm_xgb = confusion_matrix(Y_test, xgb_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ROC-AUC comparison
    from sklearn.metrics import roc_curve, auc

    fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(Y_test, xgb_prob)

    auc_rf = auc(fpr_rf, tpr_rf)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    st.subheader("📈 ROC Curve Comparison")

    fig, ax = plt.subplots()
    ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
    ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
    ax.plot([0,1], [0,1], "--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)
    plt.close()
# ══════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════
elif page == "🔮 Predict Customer":
    st.title("🔮 Predict Customer Churn")
    st.markdown("---")

    col_form, col_result = st.columns([1,1])

    with col_form:
        st.subheader("Enter Customer Details")
        tenure   = st.slider("Tenure Months", 1, 72, 12)
        monthly  = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_c  = float(tenure * monthly)
        senior   = st.selectbox("Senior Citizen", [0,1], format_func=lambda x:"Yes" if x else "No")
        partner  = st.selectbox("Partner", ["Yes","No"])
        deps     = st.selectbox("Dependents", ["Yes","No"])
        phone    = st.selectbox("Phone Service", ["Yes","No"])
        multi    = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        osec     = st.selectbox("Online Security", ["Yes","No","No internet service"])
        obak     = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        devp     = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        tech     = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        stv      = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        smov     = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paper    = st.selectbox("Paperless Billing", ["Yes","No"])
        payment  = st.selectbox("Payment Method", ["Electronic check","Mailed check",
                                                   "Bank transfer (automatic)",
                                                   "Credit card (automatic)"])
        model_c  = st.selectbox("Model", ["Random Forest","XGBoost"])
        btn      = st.button("🔮 Predict", use_container_width=True)

    with col_result:
        if btn:
            inp = pd.DataFrame([{
                'Tenure Months'    : tenure,
                'Senior Citizen'   : senior,
                'Partner'          : partner,
                'Dependents'       : deps,
                'Monthly Charges'  : monthly,
                'Total Charges'    : total_c,
                'Phone Service'    : phone,
                'Multiple Lines'   : multi,
                'Internet Service' : internet,
                'Online Security'  : osec,
                'Online Backup'    : obak,
                'Device Protection': devp,
                'Tech Support'     : tech,
                'Streaming TV'     : stv,
                'Streaming Movies' : smov,
                'Contract'         : contract,
                'Paperless Billing': paper,
                'Payment Method'   : payment
            }])
            inp_enc = pd.get_dummies(inp, drop_first=True)
            for col in X.columns:
                if col not in inp_enc.columns:
                    inp_enc[col] = 0
            inp_enc = inp_enc[X.columns]

            model = rf if model_c == "Random Forest" else xgb
            prob  = model.predict_proba(inp_enc)[0][1]

            st.subheader("Result")
            st.markdown("---")
            if prob >= 0.5:
                st.error(f"🔴 HIGH RISK\n\nChurn Probability: **{prob*100:.1f}%**")
            else:
                st.success(f"🟢 LOW RISK\n\nChurn Probability: **{prob*100:.1f}%**")

            st.markdown("---")
            st.subheader("Both Models")
            p_rf  = rf.predict_proba(inp_enc)[0][1]
            p_xgb = xgb.predict_proba(inp_enc)[0][1]
            m1,m2 = st.columns(2)
            m1.metric("Random Forest", f"{p_rf*100:.1f}%")
            m2.metric("XGBoost",       f"{p_xgb*100:.1f}%")

# ══════════════════════════════════════════════
# PAGE 3 — SEGMENTATION
# ══════════════════════════════════════════════
elif page == "🗂️ Segmentation":
    st.title("🗂️ Customer Segmentation")
    st.markdown("---")

    summary = seg_data.groupby('Cluster Segment').agg(
        Count          =('Churn Probability','count'),
        Avg_Churn_Prob =('Churn Probability','mean'),
        Avg_Tenure     =('Tenure Months','mean'),
        Avg_Monthly    =('Monthly Charges','mean')
    ).reset_index()
    summary.columns = ['Segment','Count','Avg Churn Prob','Avg Tenure','Avg Monthly']
    summary['Avg Churn Prob'] = summary['Avg Churn Prob'].map('{:.2%}'.format)
    summary['Avg Tenure']     = summary['Avg Tenure'].map('{:.1f}'.format)
    summary['Avg Monthly']    = summary['Avg Monthly'].map('${:.2f}'.format)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Tenure vs Churn Probability")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='Tenure Months',y='Churn Probability',
                        hue='Cluster Segment',data=seg_data,palette='Spectral',ax=ax)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Monthly Charges vs Churn Probability")
        fig,ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='Monthly Charges',y='Churn Probability',
                        hue='Cluster Segment',data=seg_data,palette='Spectral',ax=ax)
        st.pyplot(fig)
        plt.close()

    st.subheader("Total Charges vs Churn Probability")
    fig,ax = plt.subplots(figsize=(12,4))
    sns.scatterplot(x='Total Charges',y='Churn Probability',
                    hue='Cluster Segment',data=seg_data,palette='Spectral',ax=ax)
    st.pyplot(fig)
    plt.close()