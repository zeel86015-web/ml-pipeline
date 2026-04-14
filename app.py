import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Professional ML Pipeline Dashboard", layout="wide")
st.title("🚀 Professional ML Pipeline Dashboard")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = 'Classification'

# --- Horizontal Expansion using Tabs ---
tabs = st.tabs([
    "1. Data Input", 
    "2. EDA", 
    "3. Cleaning", 
    "4. Feature Selection", 
    "5. Split & Model", 
    "6. Train & Tune"
])

# ==========================================
# TAB 1: Setup, Data Input & PCA
# ==========================================
with tabs[0]:
    st.header("1. Problem Definition & Data Upload")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.task_type = st.selectbox("Select Problem Type:", ["Classification", "Regression"])
        uploaded_file = st.file_uploader("Upload your CSV Data", type=["csv"])
        
    if uploaded_file is not None:
        if st.session_state.df is None or st.button("Reload Original Data"):
            st.session_state.df = pd.read_csv(uploaded_file)
        
        df = st.session_state.df
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        with col2:
            target_col = st.selectbox("Select Target Feature:", df.columns.tolist(), index=len(df.columns)-1)
            st.session_state.target_col = target_col
        
        st.markdown("---")
        st.subheader("Data Shape & PCA Visualization")
        
        # PCA requires numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
            
        selected_features = st.multiselect("Select features for PCA:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        
        if len(selected_features) >= 2:
            try:
                pca = PCA(n_components=2)
                scaled_data = StandardScaler().fit_transform(df[selected_features].dropna())
                pca_result = pca.fit_transform(scaled_data)
                
                pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                pca_df['Target'] = df[target_col].dropna().values[:len(pca_df)]
                
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="2D PCA Projection of Selected Features")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not perform PCA. Ensure selected columns contain valid numeric data. Error: {e}")

# ==========================================
# TAB 2: Exploratory Data Analysis (EDA)
# ==========================================
with tabs[1]:
    st.header("2. Exploratory Data Analysis (EDA)")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
        with col2:
            st.subheader("Missing Values")
            missing_data = df.isna().sum()
            st.bar_chart(missing_data[missing_data > 0])
            if missing_data.sum() == 0:
                st.success("No missing values found!")

# ==========================================
# TAB 3: Data Engineering & Cleaning
# ==========================================
with tabs[2]:
    st.header("3. Data Engineering & Cleaning")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        # --- Imputation ---
        with col1:
            st.subheader("Missing Value Imputation")
            impute_method = st.selectbox("Select Imputation Method:", ["None", "Mean", "Median", "Mode"])
            if st.button("Apply Imputation"):
                if impute_method == "Mean":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif impute_method == "Median":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif impute_method == "Mode":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
                st.session_state.df = df
                st.success(f"{impute_method} imputation applied!")

        # --- Outlier Detection ---
        with col2:
            st.subheader("Outlier Detection")
            outlier_method = st.selectbox("Method:", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
            
            if outlier_method != "None":
                clean_df = df.dropna(subset=numeric_cols)
                outliers_mask = np.zeros(len(clean_df), dtype=bool)
                
                if outlier_method == "IQR":
                    Q1 = clean_df[numeric_cols].quantile(0.25)
                    Q3 = clean_df[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_mask = ((clean_df[numeric_cols] < (Q1 - 1.5 * IQR)) | (clean_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1).values
                    
                elif outlier_method == "Isolation Forest":
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(clean_df[numeric_cols])
                    outliers_mask = preds == -1
                    
                elif outlier_method in ["DBSCAN", "OPTICS"]:
                    scaled_data = StandardScaler().fit_transform(clean_df[numeric_cols])
                    algo = DBSCAN() if outlier_method == "DBSCAN" else OPTICS()
                    preds = algo.fit_predict(scaled_data)
                    outliers_mask = preds == -1
                    
                num_outliers = outliers_mask.sum()
                st.warning(f"Detected **{num_outliers}** outliers using {outlier_method}.")
                
                if num_outliers > 0 and st.button("Delete Outliers"):
                    # Keep only non-outliers
                    df_filtered = clean_df[~outliers_mask].reset_index(drop=True)
                    st.session_state.df = df_filtered
                    st.success("Outliers removed successfully!")

# ==========================================
# TAB 4: Feature Selection
# ==========================================
with tabs[3]:
    st.header("4. Feature Selection")
    if st.session_state.df is not None:
        df = st.session_state.df
        target_col = st.session_state.target_col
        
        # Prepare Data
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore').dropna()
        y = df.loc[X.index, target_col]
        
        st.markdown("Evaluate features against the target variable.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Variance Threshold")
            var_thresh = st.slider("Variance Threshold:", 0.0, 1.0, 0.05)
            try:
                selector = VarianceThreshold(threshold=var_thresh)
                selector.fit(X)
                kept_features = X.columns[selector.get_support()].tolist()
                st.write(f"**Features kept ({len(kept_features)}):** {kept_features}")
            except Exception as e:
                st.write("Could not apply variance threshold.")

        with col2:
            st.subheader("Information Gain")
            k_features = st.slider("Top K Features to select:", 1, len(X.columns), min(5, len(X.columns)))
            if st.button("Calculate Information Gain"):
                # Handle categorical target for classification
                if st.session_state.task_type == "Classification" and y.dtype == 'O':
                    y_encoded = LabelEncoder().fit_transform(y)
                    mi = mutual_info_classif(X, y_encoded)
                elif st.session_state.task_type == "Classification":
                    mi = mutual_info_classif(X, y)
                else:
                    mi = mutual_info_regression(X, y)
                    
                mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
                fig = px.bar(mi_series.head(k_features), orientation='h', title=f"Top {k_features} Features")
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.selected_features = mi_series.head(k_features).index.tolist()
                st.success("Features saved for model training.")

# ==========================================
# TAB 5: Data Split & Model Selection
# ==========================================
with tabs[4]:
    st.header("5. Data Split & Model Selection")
    if st.session_state.df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Train/Test Split")
            test_size = st.slider("Test Size Proportion:", 0.1, 0.5, 0.2, 0.05)
            st.session_state.test_size = test_size
            
        with col2:
            st.subheader("Model Selection")
            if st.session_state.task_type == "Classification":
                models = ["Logistic Regression", "SVM", "Random Forest", "K-Means (Unsupervised)"]
            else:
                models = ["Linear Regression", "SVR", "Random Forest", "K-Means (Unsupervised)"]
                
            selected_model = st.selectbox("Choose a Model:", models)
            st.session_state.selected_model = selected_model
            
            if "SVM" in selected_model or "SVR" in selected_model:
                st.session_state.svm_kernel = st.selectbox("SVM Kernel:", ["linear", "poly", "rbf", "sigmoid"])
            else:
                st.session_state.svm_kernel = "rbf"

# ==========================================
# TAB 6: Training, Evaluation & Tuning
# ==========================================
with tabs[5]:
    st.header("6. Training, K-Fold Validation & Metrics")
    if st.session_state.df is not None:
        df = st.session_state.df
        target_col = st.session_state.target_col
        task = st.session_state.task_type
        
        # Use selected features if available
        features = st.session_state.get('selected_features', df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore').columns.tolist())
            
        X = df[features].dropna()
        y = df.loc[X.index, target_col]
        
        # Label encode categorical target for classification
        if task == "Classification" and y.dtype == 'O':
             y = LabelEncoder().fit_transform(y)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            k_folds = st.number_input("Number of K-Folds:", 2, 20, 5)
        with col2:
            tune_method = st.selectbox("Hyperparameter Tuning:", ["None", "GridSearch", "RandomSearch"])
            
        if st.button("Train Model & Evaluate", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            sel_model = st.session_state.selected_model
            model = None
            param_grid = {}
            
            # Setup Model and Parameters
            if sel_model == "Logistic Regression":
                model = LogisticRegression()
                param_grid = {'C': [0.01, 0.1, 1, 10]}
            elif sel_model == "Linear Regression":
                model = LinearRegression()
            elif sel_model == "SVM":
                model = SVC(kernel=st.session_state.svm_kernel)
                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            elif sel_model == "SVR":
                model = SVR(kernel=st.session_state.svm_kernel)
                param_grid = {'C': [0.1, 1, 10]}
            elif sel_model == "Random Forest":
                model = RandomForestClassifier() if task == "Classification" else RandomForestRegressor()
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            elif "K-Means" in sel_model:
                model = KMeans(n_clusters=len(np.unique(y)) if len(np.unique(y)) < 20 else 3)
            
            with st.spinner("Training model..."):
                if "K-Means" in sel_model:
                    st.info("K-Means is unsupervised. Standard supervised metrics do not apply directly.")
                    model.fit(X_train_scaled)
                    st.write("Cluster Centers:", model.cluster_centers_)
                else:
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    
                    # Tuning
                    if tune_method == "GridSearch" and param_grid:
                        search = GridSearchCV(model, param_grid, cv=kf)
                        search.fit(X_train_scaled, y_train)
                        model = search.best_estimator_
                        st.success(f"Best Parameters via GridSearch: {search.best_params_}")
                    elif tune_method == "RandomSearch" and param_grid:
                        search = RandomizedSearchCV(model, param_grid, cv=kf, n_iter=5, random_state=42)
                        search.fit(X_train_scaled, y_train)
                        model = search.best_estimator_
                        st.success(f"Best Parameters via RandomSearch: {search.best_params_}")
                    else:
                        # Standard Training with Cross Validation
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)
                        st.write(f"**Mean CV Score:** {np.mean(cv_scores):.4f} (Std: {np.std(cv_scores):.4f})")
                        model.fit(X_train_scaled, y_train)

                    # Performance Metrics & Fitting Check
                    train_preds = model.predict(X_train_scaled)
                    test_preds = model.predict(X_test_scaled)
                    
                    st.markdown("### Performance Metrics")
                    m_col1, m_col2 = st.columns(2)
                    
                    if task == "Classification":
                        train_metric = accuracy_score(y_train, train_preds)
                        test_metric = accuracy_score(y_test, test_preds)
                        m_col1.metric("Training Accuracy", f"{train_metric:.4f}")
                        m_col2.metric("Testing Accuracy", f"{test_metric:.4f}")
                    else:
                        train_metric = r2_score(y_train, train_preds)
                        test_metric = r2_score(y_test, test_preds)
                        m_col1.metric("Training R²", f"{train_metric:.4f}")
                        m_col2.metric("Testing R²", f"{test_metric:.4f}")
                        st.write(f"**Testing RMSE:** {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
                    
                    # Overfitting / Underfitting Check
                    st.markdown("### Model Fitness Check")
                    if train_metric - test_metric > 0.15:
                        st.error("⚠️ **Overfitting Detected:** The model performs significantly better on training data than testing data.")
                    elif train_metric < 0.60:
                        st.warning("⚠️ **Underfitting Detected:** The model is performing poorly on the training data.")
                    else:
                        st.success("✅ **Good Fit:** The model generalized well to the unseen testing data!")

