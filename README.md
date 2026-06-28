# 📊 Customer Churn Prediction & Segmentation

Predict which telecom customers are likely to churn using Machine Learning — and segment them into groups for targeted retention strategies.

---

## 🔍 Problem Statement

A telecom company is losing customers every month. The goal is to:
- **Predict** which customers will churn before they actually leave
- **Understand** why they are churning (which features matter most)
- **Segment** customers into groups so the business can act on it

---

## 📁 Project Structure

```
churn-prediction-and-segmentation/
│
├── Customer_Segmentation_Churn_Prediction.ipynb   # EDA + Random Forest + Segmentation
├── 02_Model_Comparison_XGB_RF.ipynb               # XGBoost + Model Comparison + SHAP
├── app.py                                         # Streamlit Dashboard
├── Telco_customer_churn.xlsx                      # Dataset
└── README.md
```

---

## 📂 Dataset

- **Source:** Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Features:** 33 columns — contract type, internet service, tenure, monthly charges, etc.
- **Target:** Churn Value (0 = Stayed, 1 = Churned)
- **Churn Rate:** 26.5% (1,869 out of 7,043 customers churned)

---

## ⚙️ Tech Stack

| Tool | Use |
|------|-----|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualizations |
| Scikit-learn | Random Forest, KMeans, preprocessing |
| XGBoost | Gradient boosting model |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |

---

## 🤖 Models Used

### Random Forest
- Handled class imbalance using `class_weight='balanced'`
- Hyperparameter tuning — `n_estimators=300`, `max_depth=10`
- Feature importance analysis
- Combination loop — tried different trees and depths
- Cross validation (cv=5)

### XGBoost
- Handled class imbalance using `scale_pos_weight=3`
- Hyperparameter tuning — `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`
- Feature importance analysis
- Combination loop — tried different trees and depths
- Cross validation (cv=5)

---

## 📊 Results

| Model | CV Accuracy | CV Recall | AUC-ROC |
|-------|------------|-----------|---------|
| Random Forest | 78.30% | 72.66% | 85.71% |
| XGBoost | 76.19% | 77.15% | 85.16% |

> **Why Recall matters most here:**
> A missed churner = lost customer = lost revenue.
> XGBoost has higher Recall — it catches more real churners.
> Random Forest has higher AUC-ROC — it separates churners better overall.

---

## 🧠 SHAP Explainability

SHAP explains **why** the model made a prediction — it shows how much each feature pushed the churn probability up or down.

**Top features driving churn (Random Forest):**
1. Tenure Months — longer tenure = less likely to churn
2. Contract Two Year — two year contract = much less churn
3. Internet Service Fiber Optic — fiber users churn more
4. Dependents — customers with dependents are more stable
5. Monthly Charges — higher charges = higher churn risk

---

## 🗂️ Customer Segmentation

Used **KMeans clustering (k=3)** on Tenure, Monthly Charges, Total Charges, and Churn Probability.

| Segment | Description | Avg Churn Risk |
|---------|-------------|---------------|
| Budget Loyal Customers | Long tenure, low charges, stable | Low |
| High Risk New Customers | Short tenure, high churn probability | High |
| Loyal Premium Customers | Mid tenure, higher charges, moderate risk | Medium |

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AayushiSharma2005/churn-prediction-and-segmentation.git
cd churn-prediction-and-segmentation
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost shap openpyxl
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📱 Dashboard Pages

| Page | What it shows |
|------|--------------|
| 🏠 Overview | Churn rate, KPIs, charts by contract and internet service |
| 🔮 Predict Customer | Fill customer details → instant churn probability from RF and XGBoost |
| 🗂️ Segmentation | 3 customer clusters with scatter plots |

---

## 💡 Key Insights

- Customers on **Month-to-month contracts** churn the most (42.7%)
- **Fiber optic** internet users have higher churn than DSL users
- Customers with **longer tenure** are much less likely to churn
- **Electronic check** payment users churn more than auto-pay users
- Customers with **Two year contracts** have only 2.8% churn rate

---

*Made by Aayushi Sharma*
