import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, OPTICS
import io

# Set page config for a premium feel
st.set_page_config(
    page_title="ML Intelligence Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for glassmorphism and premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
        color: #f8fafc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4);
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    .highlight {
        color: #3b82f6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for session state initialization
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = "Classification"
    if 'features' not in st.session_state:
        st.session_state.features = []
    if 'steps_completed' not in st.session_state:
        st.session_state.steps_completed = 0

def main():
    init_session_state()

    st.title("🚀 NexuStream ML Intelligence Pipeline")
    st.title("🚀 ML Intelligence Pipeline")
    st.markdown("Transform raw data into predictive insights with our state-of-the-art automated pipeline.")

    # Horizontal Step-based Navigation using Tabs
    tabs = st.tabs([
        "🎯 Problem Type", 
        "📥 Data Input", 
        "📊 EDA", 
        "🛠️ Data Cleaning", 
        "🧬 Feature Selection", 
        "✂️ Data Split", 
        "🤖 Model Selection", 
        "⚙️ Training & Validation",
        "📈 Metrics & Tuning"
    ])

    # --- TAB 1: PROBLEM TYPE ---
    with tabs[0]:
        st.header("Select Your Objective")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Classification", use_container_width=True):
                st.session_state.problem_type = "Classification"
                st.success("Target set to Classification")
        with col2:
            if st.button("Regression", use_container_width=True):
                st.session_state.problem_type = "Regression"
                st.success("Target set to Regression")

        st.info(f"Current selection: **{st.session_state.problem_type}**")

    # --- TAB 2: DATA INPUT ---
    with tabs[1]:
        st.header("Upload Dataset")

        # Sample Data Option
        col_s1, col_s2 = st.columns([1, 4])
        with col_s1:
            if st.button("Load Sample Data (Titanic)", use_container_width=True):
                # Using a generic URL for titanic or iris
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                st.session_state.data = pd.read_csv(url)
                st.session_state.problem_type = "Classification"
                st.success("Sample data loaded!")

        # Local File Path Option
        local_path = st.text_input("Or enter local file path (e.g., /home/akshonite/Downloads/tmdb_5000_credits.csv)")
        if st.button("Load from Path"):
            try:
                if local_path.endswith('.csv'):
                    st.session_state.data = pd.read_csv(local_path)
                else:
                    st.session_state.data = pd.read_excel(local_path)
                st.success(f"Data loaded from {local_path}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        uploaded_file = st.file_uploader("Or Browse your own CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file or st.session_state.data is not None:
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.data = pd.read_csv(uploaded_file)
                    else:
                        st.session_state.data = pd.read_excel(uploaded_file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error processing upload: {e}")

            df = st.session_state.data
            st.dataframe(df.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.session_state.target = st.selectbox("Select Target Feature", df.columns)
            with col2:
                features = [col for col in df.columns if col != st.session_state.target]
                st.session_state.features = st.multiselect("Select Input Features", features, default=features)

            if st.session_state.features:
                st.subheader("Data Visualizations")

                # PCA Visualization
                numeric_df = df[st.session_state.features].select_dtypes(include=[np.number]).dropna()
                if not numeric_df.empty and len(st.session_state.features) >= 2:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_df)
                    pca = PCA(n_components=min(3, len(numeric_df.columns)))
                    pca_result = pca.fit_transform(scaled_data)

                    if pca_result.shape[1] >= 2:
                        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

                        # Add target for coloring if numeric or low cardinality
                        if df[st.session_state.target].nunique() < 20:
                             pca_df['Target'] = df.loc[numeric_df.index, st.session_state.target].values
                             fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="2D PCA Visualization", template="plotly_dark")
                        else:
                             fig = px.scatter(pca_df, x='PC1', y='PC2', title="2D PCA Visualization", template="plotly_dark")

                        st.plotly_chart(fig, use_container_width=True)

                    st.write(f"Overall Data Shape: {df.shape}")
                    st.write(f"Selected Features Sub-shape: {df[st.session_state.features].shape}")
                else:
                    st.warning("Please select at least 2 numeric features for PCA visualization.")

    # --- TAB 3: EDA ---
    with tabs[2]:
        if st.session_state.data is not None:
            st.header("Exploratory Data Analysis")
            df = st.session_state.data

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isna().sum().sum())

            tab_corr, tab_dist, tab_stats = st.tabs(["Correlation Matrix", "Distributions", "Summary Stats"])

            with tab_corr:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Not enough numeric columns for correlation.")

            with tab_dist:
                feature_to_plot = st.selectbox("Choose feature to visualize distribution", df.columns)
                fig = px.histogram(df, x=feature_to_plot, color=st.session_state.target if df[st.session_state.target].nunique() < 10 else None, 
                                   marginal="box", title=f"Distribution of {feature_to_plot}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with tab_stats:
                st.table(df.describe().T)
        else:
            st.warning("Please upload data first.")

    # --- TAB 4: DATA ENGINEERING & CLEANING ---
    with tabs[3]:
        if st.session_state.data is not None:
            st.header("Data Cleaning & Engineering")
            df = st.session_state.data.copy()

            st.subheader("1. Imputation (Handling Missing Values)")
            strategy = st.radio("Imputation Strategy", ["Mean", "Median", "Mode", "Constant (0)"], horizontal=True)

            if st.button("Apply Imputation"):
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype in [np.float64, np.int64]:
                            if strategy == "Mean":
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif strategy == "Median":
                                df[col].fillna(df[col].median(), inplace=True)
                            elif strategy == "Mode":
                                df[col].fillna(df[col].mode()[0], inplace=True)
                            else:
                                df[col].fillna(0, inplace=True)
                        else:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                st.session_state.data = df
                st.success("Missing values imputed.")

            st.divider()

            st.subheader("2. Outlier Detection")
            method = st.selectbox("Outlier Detection Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])

            outliers_found = pd.Series([False] * len(df))

            numeric_df = df.select_dtypes(include=[np.number])
            if method == "IQR":
                Q1 = numeric_df.quantile(0.25)
                Q3 = numeric_df.quantile(0.75)
                IQR = Q3 - Q1
                outliers_found = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            elif method == "Isolation Forest":
                clf = IsolationForest(contamination=0.05, random_state=42)
                preds = clf.fit_predict(numeric_df.fillna(0))
                outliers_found = (preds == -1)
            elif method == "DBSCAN":
                db = DBSCAN(eps=3, min_samples=2)
                preds = db.fit_predict(StandardScaler().fit_transform(numeric_df.fillna(0)))
                outliers_found = (preds == -1)
            elif method == "OPTICS":
                opt = OPTICS(min_samples=5)
                preds = opt.fit_predict(StandardScaler().fit_transform(numeric_df.fillna(0)))
                outliers_found = (preds == -1)

            num_outliers = outliers_found.sum()
            st.write(f"Detected **{num_outliers}** outliers.")

            if num_outliers > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Remove Outliers"):
                        st.session_state.data = df[~outliers_found]
                        st.success(f"Removed {num_outliers} outliers.")
                        st.rerun()
                with col2:
                    if st.button("Keep Outliers"):
                        st.info("Outliers retained.")

            st.subheader("Quick Data Delete")
            rows_to_drop = st.multiselect("Select Row Indices to Delete", df.index.tolist())
            if st.button("Delete Selected Rows"):
                st.session_state.data = df.drop(index=rows_to_drop)
                st.success(f"Deleted {len(rows_to_drop)} rows.")
                st.rerun()

        else:
            st.warning("Please upload data first.")

    # --- TAB 5: FEATURE SELECTION ---
    with tabs[4]:
        if st.session_state.data is not None:
            st.header("Feature Selection")
            df = st.session_state.data
            y = df[st.session_state.target]
            X = df[st.session_state.features].select_dtypes(include=[np.number])

            # Encode target if classification
            if st.session_state.problem_type == "Classification" and y.dtype == object:
                y = LabelEncoder().fit_transform(y.astype(str))

            st.write("Only numeric features are considered for automated selection.")

            # 1. Variance Threshold
            st.subheader("1. Variance Threshold")
            threshold = st.slider("Min Variance", 0.0, 1.0, 0.0, 0.05)
            if st.button("Apply Variance Filter"):
                vt = VarianceThreshold(threshold=threshold)
                vt.fit(X)
                selected_vars = X.columns[vt.get_support()].tolist()
                st.write(f"Selected Features: {selected_vars}")
                st.session_state.features = selected_vars

            # 2. Correlation Filter
            st.subheader("2. Correlation with Target")
            corr_thresh = st.slider("Min Absolute Correlation with Target", 0.0, 1.0, 0.1, 0.05)
            if st.button("Apply Correlation Filter"):
                # Join X and y for correlation
                temp_df = X.copy()
                temp_df['__target__'] = y
                correlations = temp_df.corr()['__target__'].abs().sort_values(ascending=False)
                selected_corr = correlations[correlations >= corr_thresh].index.tolist()
                selected_corr = [c for c in selected_corr if c != '__target__']
                st.session_state.features = selected_corr
                st.write(f"Selected Features: {selected_corr}")

            # 3. Information Gain / Mutual Info
            st.subheader("3. Information Gain")
            if not X.empty and len(X.columns) > 0:
                k_max = len(X.columns)
                k_default = min(5, k_max)
                k_best = st.number_input("Select Top K Features", 1, k_max, k_default)
                if st.button("Apply Mutual Information"):
                    with st.spinner("Calculating Mutual Information..."):
                        if st.session_state.problem_type == "Classification":
                            mi = mutual_info_classif(X.fillna(0), y)
                        else:
                            mi = mutual_info_regression(X.fillna(0), y)

                        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
                        selected_mi = mi_series.head(k_best).index.tolist()
                        st.session_state.features = selected_mi
                        st.write(f"Selected Features: {selected_mi}")

                        fig = px.bar(mi_series.head(k_best), title="Top Features by Information Gain", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric features available for Mutual Information calculation.")

        else:
            st.warning("Please upload data first.")

    # --- TAB 6: DATA SPLIT ---
    with tabs[5]:
         if st.session_state.data is not None:
            st.header("Train-Test Split")
            test_size = st.slider("Test Size (%)", 10, 50, 20)
            random_state = st.number_input("Random State", 0, 100, 42)

            if st.button("Perform Split"):
                df = st.session_state.data
                X = df[st.session_state.features]
                y = df[st.session_state.target]

                # Preprocessing
                # Simple handling of categorical in X
                X_proc = pd.get_dummies(X)

                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                    X_proc, y, test_size=test_size/100, random_state=random_state
                )

                st.success(f"Data split completed. Train: {st.session_state.X_train.shape[0]}, Test: {st.session_state.X_test.shape[0]}")
                st.session_state.steps_completed = 6
         else:
            st.warning("Please finalize features before splitting.")

    # --- TAB 7: MODEL SELECTION ---
    with tabs[6]:
        st.header("Choose Your Model")

        if st.session_state.problem_type == "Classification":
            model_options = ["Random Forest", "SVM", "Logistic Regression", "K-Means"]
        else:
            model_options = ["Random Forest", "SVM", "Linear Regression", "K-Means"]

        st.session_state.selected_model = st.selectbox("Model", model_options)

        if st.session_state.selected_model == "SVM":
            st.session_state.svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        elif st.session_state.selected_model == "K-Means":
            st.session_state.km_clusters = st.slider("Number of Clusters", 2, 10, 3)
            st.info("Note: K-Means is primarily unsupervised, but can be used here for cluster analysis.")

    # --- TAB 8: TRAINING & VALIDATION ---
    with tabs[7]:
        if 'X_train' in st.session_state:
            st.header("Model Training & K-Fold Validation")
            k_fold = st.number_input("Value of K (Cross Validation)", 2, 10, 5)

            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    # Initialize Model
                    if st.session_state.problem_type == "Classification":
                        if st.session_state.selected_model == "Random Forest":
                            model = RandomForestClassifier()
                        elif st.session_state.selected_model == "SVM":
                            model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'))
                        elif st.session_state.selected_model == "Logistic Regression":
                            model = LogisticRegression()
                        else:
                            model = KMeans(n_clusters=st.session_state.get('km_clusters', 3))
                    else:
                        if st.session_state.selected_model == "Random Forest":
                            model = RandomForestRegressor()
                        elif st.session_state.selected_model == "SVM":
                            model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
                        elif st.session_state.selected_model == "Linear Regression":
                            model = LinearRegression()
                        else:
                            model = KMeans(n_clusters=st.session_state.get('km_clusters', 3))

                    # Encode target if necessary
                    y_train = st.session_state.y_train
                    if st.session_state.problem_type == "Classification" and y_train.dtype == object:
                        le = LabelEncoder()
                        y_train = le.fit_transform(y_train.astype(str))
                        st.session_state.label_encoder = le

                    # Fit and Cross Validate
                    if st.session_state.selected_model != "K-Means":
                        cv_results = cross_validate(model, st.session_state.X_train, y_train, cv=k_fold, return_train_score=True)
                        model.fit(st.session_state.X_train, y_train)
                        st.session_state.trained_model = model

                        st.success("Training Complete!")
                        st.subheader("K-Fold Results")
                        cv_df = pd.DataFrame(cv_results)
                        st.dataframe(cv_df, use_container_width=True)

                        # Store for metrics
                        st.session_state.cv_train_mean = cv_results['train_score'].mean()
                        st.session_state.cv_test_mean = cv_results['test_score'].mean()
                    else:
                        model.fit(st.session_state.X_train)
                        st.session_state.trained_model = model
                        st.success("Clustering Complete!")
        else:
            st.warning("Please split data first.")

    # --- TAB 9: METRICS & HYPERPARAMETER TUNING ---
    with tabs[8]:
        if 'trained_model' in st.session_state:
            st.header("Performance Metrics & Tuning")

            # 1. Metrics
            st.subheader("1. Evaluation Metrics")
            model = st.session_state.trained_model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            if st.session_state.problem_type == "Classification" and hasattr(st.session_state, 'label_encoder'):
                y_test = st.session_state.label_encoder.transform(y_test.astype(str))

            if st.session_state.selected_model != "K-Means":
                preds = model.predict(X_test)

                if st.session_state.problem_type == "Classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='weighted')
                    st.metric("Test Accuracy", f"{acc:.4f}")
                    st.metric("F1 Score (Weighted)", f"{f1:.4f}")
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, preds))
                else:
                    mse = mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    st.metric("Test MSE", f"{mse:.4f}")
                    st.metric("Test R2 Score", f"{r2:.4f}")

                # Underfitting/Overfitting Check
                st.subheader("Overfitting/Underfitting Diagnosis")
                train_score = st.session_state.cv_train_mean
                test_score = st.session_state.cv_test_mean
                st.write(f"Mean Train CV Score: {train_score:.4f}")
                st.write(f"Mean Test CV Score: {test_score:.4f}")

                diff = train_score - test_score
                if diff > 0.15:
                    st.error("⚠️ The model is likely **OVERFITTING**. (Large gap between train and test)")
                elif train_score < 0.6:
                    st.warning("⚠️ The model is likely **UNDERFITTING**. (Low train performance)")
                else:
                    st.success("✅ The model seems **WELL-BALANCED**.")

            # 2. Hyperparameter Tuning
            st.divider()
            st.subheader("2. Hyperparameter Tuning")
            search_type = st.selectbox("Search Method", ["Grid Search", "Random Search"])

            if st.session_state.selected_model == "Random Forest":
                 param_grid = {
                     'n_estimators': [50, 100, 200],
                     'max_depth': [None, 10, 20],
                     'min_samples_split': [2, 5]
                 }
            elif st.session_state.selected_model == "SVM":
                 param_grid = {
                     'C': [0.1, 1, 10],
                     'gamma': ['scale', 'auto']
                 }
            else:
                 param_grid = {}
                 st.write("Tuning options limited for this model.")

            if param_grid and st.button("Run Hyperparameter Optimization"):
                with st.spinner("Optimizing..."):
                    if search_type == "Grid Search":
                        search = GridSearchCV(model, param_grid, cv=3)
                    else:
                        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3)

                    search.fit(st.session_state.X_train, st.session_state.y_train)
                    st.write("Best Parameters:", search.best_params_)
                    st.write("Best Score:", search.best_score_)

                    # Store original performance
                    original_score = st.session_state.cv_test_mean
                    improvement = search.best_score_ - original_score
                    st.metric("Improvement", f"{improvement:.4f}", delta=f"{improvement:.4f}")

        else:
            st.warning("Please train the model first.")

if __name__ == "__main__":
    main()