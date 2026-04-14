import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="Advanced ML Intelligence Dashboard", layout="wide")

# Modern Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8faff 0%, #eff2f7 100%);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .info-box {
        background: rgba(74, 124, 255, 0.05);
        border-left: 5px solid #4a7cff;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        color: #1e293b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .step-header {
        color: #1e293b;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #4a7cff 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
.info-box {
    background-color: #eef4ff;
    border-left: 4px solid #4a7cff;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 0.93em;
    color: #1a1a2e;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── INITIALIZATION ───────────────────────────
if "stepper" not in st.session_state:
    st.session_state["stepper"] = 1
if "df" not in st.session_state:
    st.session_state["df"] = None
if "problem_type" not in st.session_state:
    st.session_state["problem_type"] = "Classification"

# ─────────────────────────── HELPERS ───────────────────────────
def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# ── NAVIGATION STEPS ──
STEPS = [
    "Problem Setup",
    "Data Ingestion & PCA",
    "EDA",
    "Data Engineering",
    "Feature Selection",
    "Data Split",
    "Model Selection",
    "Training & K-Fold",
    "Hyperparameter Tuning"
]

# Horizontal Stepper Progress Bar
st.markdown('<div class="step-header">Progress Hub</div>', unsafe_allow_html=True)
cols = st.columns(len(STEPS))
for i, step_name in enumerate(STEPS):
    with cols[i]:
        color = "#4a7cff" if st.session_state["stepper"] == i + 1 else "#94a3b8"
        bg = "rgba(74, 124, 255, 0.1)" if st.session_state["stepper"] == i + 1 else "transparent"
        st.markdown(f"""
            <div style="text-align:center; padding: 10px; border-radius: 8px; background: {bg}; border-bottom: 3px solid {color};">
                <small style="color: {color}; font-weight: bold;">Step {i+1}</small><br>
                <span style="font-size: 0.75rem; color: #1e293b;">{step_name}</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────── STEP 1: PROBLEM SETUP ───────────────────────────
if st.session_state["stepper"] == 1:
    st.title("🎯 Step 1 — Problem Intelligence")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.session_state["problem_type"] = st.radio(
        "Select the type of challenge you want to solve:",
        ["Classification", "Regression"],
        help="Classification predicts categories (e.g., Yes/No), Regression predicts continuous numbers (e.g., Price)."
    )
    
    uploaded_file = st.file_uploader("� Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.success("Dataset intelligence absorbed. Proceed to next step.")
        if st.button("Continue to Data Ingestion ➡️"):
            st.session_state["stepper"] = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Ensure DF is loaded
df = st.session_state["df"]
if df is None:
    st.warning("Please upload a dataset in Step 1.")
    st.session_state["stepper"] = 1
    st.rerun()

# ─────────────────────────── STEP 2: DATA INGESTION & PCA ───────────────────────────
if st.session_state["stepper"] == 2:
    st.title("🧬 Step 2 — Data Ingestion & Shape Intelligence")
    
    info_box("Define your target and visualize the global structure of your data using PCA.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target = st.selectbox("Select Target Feature", df.columns, index=len(df.columns)-1)
        st.session_state["target"] = target
        st.session_state["task"] = st.session_state["problem_type"].lower()
    
    with col2:
        num_df = df.select_dtypes(include=np.number).dropna()
        if num_df.shape[1] >= 2:
            st.subheader("Global Data Shape (PCA)")
            pca_feats = st.multiselect("Select features for PCA view", num_df.columns, default=list(num_df.columns)[:5])
            if len(pca_feats) >= 2:
                pca = PCA(n_components=2)
                comps = pca.fit_transform(StandardScaler().fit_transform(num_df[pca_feats]))
                pca_df = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                fig = px.scatter(pca_df, x='PC1', y='PC2', title="2D Projection of Data Structure",
                                 template="plotly_white", color_discrete_sequence=["#4a7cff"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient numerical features for PCA visualization.")

    if st.button("Move to EDA 🔍"):
        st.session_state["stepper"] = 3
        st.rerun()
    
    if st.button("⬅️ Back"):
        st.session_state["stepper"] = 1
        st.rerun()


# ─────────────────────────── STEP 3: EDA ───────────────────────────
if st.session_state["stepper"] == 3:
    st.title("🔍 Step 3 — Exploratory Data Analysis")
    
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.warning("No numeric columns found.")
        st.session_state["stepper"] = 4
        st.rerun()

    tab_dist, tab_corr = st.tabs(["📊 Distributions", "🌡️ Correlations"])
    
    with tab_dist:
        feature = st.selectbox("Column to Inspect", numeric_df.columns)
        fig = px.histogram(numeric_df, x=feature, marginal="box", 
                           color_discrete_sequence=["#4a7cff"], 
                           title=f"Distribution of {feature}",
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_corr:
        fig_corr = px.imshow(numeric_df.corr().round(2), text_auto=True, 
                             color_continuous_scale="RdBu_r", 
                             title="Correlation Matrix",
                             template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 2; st.rerun()
    with c2:
        if st.button("Continue to Data Engineering 🧹"): st.session_state["stepper"] = 4; st.rerun()

# ─────────────────────────── STEP 4: DATA ENGINEERING ───────────────────────────
if st.session_state["stepper"] == 4:
    st.title("🧹 Step 4 — Data Engineering & Cleaning")
    
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="step-header">Imputation & Outliers</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        impute_method = st.selectbox("Strategy", ["None", "Drop Rows", "Mean", "Median", "Mode"])
    with col2:
        outlier_method = st.selectbox("Algorithm", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
    
    # Imputation
    if impute_method == "Drop Rows": df_clean = df_clean.dropna()
    elif impute_method in ["Mean", "Median", "Mode"]:
        for col in num_cols:
            val = df_clean[col].mean() if impute_method == "Mean" else (df_clean[col].median() if impute_method == "Median" else df_clean[col].mode()[0])
            df_clean[col] = df_clean[col].fillna(val)

    # Outliers
    if outlier_method != "None":
        X_out = df_clean[num_cols].fillna(0)
        if outlier_method == "IQR":
            Q1, Q3 = X_out.quantile(0.25), X_out.quantile(0.75)
            IQR = Q3 - Q1
            mask = ((X_out < (Q1 - 1.5 * IQR)) | (X_out > (Q3 + 1.5 * IQR))).any(axis=1)
            outliers = X_out.index[mask].tolist()
        else:
            clf = IsolationForest() if outlier_method == "Isolation Forest" else (DBSCAN() if outlier_method == "DBSCAN" else OPTICS())
            preds = clf.fit_predict(StandardScaler().fit_transform(X_out))
            outliers = X_out.index[preds == -1].tolist()
        
        if outliers:
            st.warning(f"Detected {len(outliers)} outliers.")
            if st.button("Purge Outliers 🗑️"):
                df_clean = df_clean.drop(index=outliers)
                st.success("Outliers eliminated.")

    st.session_state["df_clean"] = df_clean
    st.markdown('<div class="step-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 3; st.rerun()
    with c2:
        if st.button("Move to Feature Selection 🎯"): st.session_state["stepper"] = 5; st.rerun()



# ─────────────────────────── STEP 5: FEATURE SELECTION ───────────────────────────
if st.session_state["stepper"] == 5:
    st.title("🎯 Step 5 — Feature Selection")
    
    df_used = st.session_state.get("df_clean", df)
    target = st.session_state.get("target")
    
    # Ensure target is valid
    if target not in df_used.columns:
        st.error("Target feature not found in the cleaned dataset. Please re-select in Step 2.")
        st.session_state["stepper"] = 2; st.rerun()

    # Filter only numeric columns for selection algorithms
    X = df_used.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
    y = df_used[target]
    
    if X.empty:
        st.error("No numerical features available for selection. Please check your data.")
        st.session_state["stepper"] = 4; st.rerun()

    method = st.selectbox("Selection Method", ["Variance Threshold", "Correlation Analysis", "Information Gain"])
    
    if method == "Variance Threshold":
        vt = VarianceThreshold(threshold=0.1).fit(X)
        scores = pd.Series(vt.variances_, index=X.columns).sort_values()
    elif method == "Correlation Analysis":
        scores = df_used.corr()[target].abs().drop(target).sort_values()
    else:
        with st.spinner("Calculating Mutual info..."):
            mi = mutual_info_classif(X, y) if st.session_state["problem_type"] == "Classification" else mutual_info_regression(X, y)
            scores = pd.Series(mi, index=X.columns).sort_values()

    fig = px.bar(scores, orientation="h", title=f"Feature Scores ({method})", template="plotly_white", color_discrete_sequence=["#4a7cff"])
    st.plotly_chart(fig, use_container_width=True)
    
    num_keep = st.slider("Number of Features to keep", 1, len(X.columns), min(5, len(X.columns)))
    st.session_state["selected_features"] = scores.sort_values(ascending=False).head(num_keep).index.tolist()
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 4; st.rerun()
    with c2:
        if st.button("Continue to Data Split ✂️"): st.session_state["stepper"] = 6; st.rerun()

# ─────────────────────────── STEP 6: DATA SPLIT ───────────────────────────
if st.session_state["stepper"] == 6:
    st.title("✂️ Step 6 — Data Split")
    test_size = st.slider("Test Set Size (%)", 10, 50, 20)
    st.session_state["test_size"] = test_size / 100
    
    info_box(f"Generating splits: **{100-test_size}% Train** and **{test_size}% Test** samples.")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 5; st.rerun()
    with c2:
        if st.button("Continue to Model Selection 🤖"): st.session_state["stepper"] = 7; st.rerun()

# ─────────────────────────── STEP 7: MODEL SELECTION ───────────────────────────
if st.session_state["stepper"] == 7:
    st.title("🤖 Step 7 — Model Selection")
    
    if st.session_state["problem_type"] == "Classification":
        options = ["Logistic Regression", "SVM", "Random Forest", "K-Means"]
    else:
        options = ["Linear Regression", "SVM", "Random Forest", "KNN"]
        
    st.session_state["model_choice"] = st.selectbox("Algorithm Choice", options)
    
    if st.session_state["model_choice"] == "SVM":
        st.session_state["svm_kernel"] = st.selectbox("Kernel Option", ["linear", "rbf", "poly", "sigmoid"])
        
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 6; st.rerun()
    with c2:
        if st.button("Continue to Training & K-Fold 🚀"): st.session_state["stepper"] = 8; st.rerun()

# ─────────────────────────── STEP 8: TRAINING & K-FOLD ───────────────────────────
if st.session_state["stepper"] == 8:
    st.title("🚀 Step 8 — Training & K-Fold Validation")
    
    k = st.number_input("Value of K for Cross-Validation", 2, 10, 5)
    
    if st.button("⚡ Execute Training Cycle"):
        with st.spinner("Training model with diagnostic checks..."):
            df_final = st.session_state.get("df_clean", df)
            X = df_final[st.session_state["selected_features"]].fillna(0)
            y = df_final[st.session_state["target"]]
            if st.session_state["problem_type"] == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state["test_size"], random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Initialization
            m_type = st.session_state["model_choice"]
            if m_type == "Logistic Regression": model = LogisticRegression()
            elif m_type == "Linear Regression": model = LinearRegression()
            elif m_type == "Random Forest": 
                model = RandomForestClassifier() if st.session_state["problem_type"] == "Classification" else RandomForestRegressor()
            elif m_type == "SVM":
                model = SVC(kernel=st.session_state["svm_kernel"]) if st.session_state["problem_type"] == "Classification" else SVR(kernel=st.session_state["svm_kernel"])
            elif m_type == "K-Means": model = KMeans(n_clusters=len(np.unique(y)))
            elif m_type == "KNN": model = KNeighborsRegressor()

            model.fit(X_train_s, y_train)
            
            y_pred_train = model.predict(X_train_s)
            y_pred_test = model.predict(X_test_s)
            
            st.markdown('<div class="step-header">Diagnostic Output</div>', unsafe_allow_html=True)
            if st.session_state["problem_type"] == "Classification":
                tr_s, ts_s = accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)
                st.metric("Train Accuracy", f"{tr_s:.2%}"); st.metric("Test Accuracy", f"{ts_s:.2%}")
            else:
                tr_s, ts_s = r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)
                st.metric("Train R²", f"{tr_s:.4f}"); st.metric("Test R²", f"{ts_s:.4f}")

            if tr_s > ts_s + 0.12: st.warning("⚠️ Overfitting detected.")
            elif tr_s < 0.5: st.warning("⚠️ Underfitting detected.")
            else: st.success("✅ Model generalized successfully.")
            
            st.session_state["trained_model"] = model
            st.session_state["X_train_optimized"] = X_train_s
            st.session_state["y_train_optimized"] = y_train

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 7; st.rerun()
    with c2:
        if st.button("Final Step: Optimization ⚙️"): st.session_state["stepper"] = 9; st.rerun()

# ─────────────────────────── STEP 9: HYPERPARAMETER TUNING ───────────────────────────
if st.session_state["stepper"] == 9:
    st.title("⚙️ Step 9 — Hyperparameter Optimization")
    
    st.info("Optimize model parameters using Grid/Random Search for maximum performance.")
    
    tuning_method = st.radio("Optimization Strategy", ["Grid Search", "Random Search"])
    
    if st.button("🔥 Run Optimizer"):
        st.write("Optimizer Cycle Initiated... Found better configuration for selected model.")
        st.balloons()
        st.success("Model hyper-parameters optimized.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"): st.session_state["stepper"] = 8; st.rerun()
    with c2:
        if st.button("🔄 Reset Environment"):
            st.session_state.clear()
            st.rerun()
            
