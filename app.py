import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

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

def info_box(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

# ─────────────────────────── HELPERS ───────────────────────────
def detect_task_type(series, threshold=10):
    if series.dtype == object or str(series.dtype) == "category":
        return "classification"
    return "classification" if series.nunique() <= threshold else "regression"

def encode_target(y):
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), name=y.name), le
    return y, None

def get_per_col_outliers(df_numeric):
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    counts = {}
    for col in df_numeric.columns:
        mask = (df_numeric[col] < (Q1[col] - 1.5 * IQR[col])) | \
               (df_numeric[col] > (Q3[col] + 1.5 * IQR[col]))
        counts[col] = int(mask.sum())
    return pd.Series(counts)

# ─────────────────────────── SIDEBAR ───────────────────────────
st.sidebar.title("🛠️ ML Workspace")
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded ✅")
else:
    st.title("👋 Welcome to the ML Pipeline Dashboard")
    st.markdown("""
    Upload a **CSV file** using the sidebar on the left to begin your machine learning workflow.

    **The pipeline covers:**
    1. 📊 **Dashboard** — Dataset overview & statistics
    2. 🔍 **EDA** — Distributions & correlations
    3. 🧹 **Data Cleaning** — Handle missing values & outliers
    4. 🎯 **Feature Selection** — Pick the most useful columns
    5. 🤖 **Model Training** — Train, evaluate & validate your model
    """)
    st.stop()

page = st.sidebar.selectbox("Navigate", [
    "Dashboard",
    "EDA",
    "Data Cleaning",
    "Feature Selection",
    "Model Training",
])

# ═══════════════════════════ DASHBOARD ═══════════════════════════
if page == "Dashboard":
    st.title("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)


# ═══════════════════════════ EDA ═══════════════════════════
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.warning("No numeric columns found in this dataset.")
        st.stop()

    st.subheader("Distribution Plot")
    info_box(
        "Select any numeric column to see how its values are spread out. "
        "The histogram shows frequency of value ranges; the box plot above shows median, quartiles, and outliers."
    )
    feature = st.selectbox("Select a column to inspect", numeric_df.columns)
    fig = px.histogram(numeric_df, x=feature, marginal="box",
                       color_discrete_sequence=["#FF4B4B"],
                       title=f"Distribution of {feature}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    info_box(
        "Correlation measures how strongly two columns move together. "
        "Values near +1 = strong positive relationship, near -1 = strong negative, near 0 = little relationship."
    )
    fig2 = px.imshow(numeric_df.corr().round(2), text_auto=True,
                     color_continuous_scale="RdBu_r", aspect="auto",
                     title="Correlation Heatmap")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════ DATA CLEANING ═══════════════════════════
elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning")

    info_box(
        "Before training any model, we need clean data. "
        "This step handles missing values, outliers, and invalid zero values (like 0 glucose, 0 BMI)."
    )

    numeric_df_raw = df.select_dtypes(include=np.number)



        # ───── HARD CODED INVALID ZERO RULES ─────
    invalid_zero_columns = []

    cols = list(df.columns)

    # Diabetes dataset
    if "Glucose" in cols and "BMI" in cols:
        invalid_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Titanic dataset
    elif "Survived" in cols and "Pclass" in cols:
        invalid_zero_columns = ["Age"]

    # Housing dataset
    elif "RM" in cols and "LSTAT" in cols:
        invalid_zero_columns = ["RM"]
    

    # # ── ZERO VALUE DETECTION ──
    # st.subheader("⚠️ Biologically Invalid Zero Values Detection")

    # zero_counts = {}
    # for col in numeric_df_raw.columns:
    #     zero_counts[col] = int((numeric_df_raw[col] == 0).sum())
    # suggested_invalid_cols = []

    # for col in numeric_df_raw.columns:
    #     zero_count = int((numeric_df_raw[col] == 0).sum())
    #     zero_counts[col] = zero_count

    # # Hardcoded priority
    #     if col in invalid_zero_columns:
    #         suggested_invalid_cols.append(col)


    
    # zero_series = pd.Series(zero_counts)
    # zero_series = zero_series[zero_series > 0]

    # if zero_series.empty:
    #     st.success("No suspicious zero values found!")
    # else:
    #     st.dataframe(
    #         zero_series.rename("Zero Count")
    #                    .reset_index()
    #                    .rename(columns={"index": "Column"}),
    #         use_container_width=True
    #     )



    

    # # ── Before-cleaning stats ──
    # st.subheader("Before Cleaning — Current Data Issues")

    # col_mv, col_out = st.columns(2)
            # ── Before-cleaning stats ──
    # st.subheader("Before Cleaning — Current Data Issues")
    st.markdown(
    "<h3 style='text-align: center; color:red;'>Before Cleaning — Current Data Issues</h3>",
    unsafe_allow_html=True
    )
    col_zero, col_mv, col_out = st.columns(3)
    
    # ───────── ZERO VALUES ─────────
    with col_zero:
        st.markdown("**Invalid Zero Values**")
        
        # zero_series = pd.Series(zero_counts)
        # zero_series = zero_series[zero_series > 0]
    
        # if zero_series.empty:
        #     st.success("No zero issues")
        # else:
        #     st.dataframe(
        #         zero_series.rename("Zero Count")
        #                    .reset_index()
        #                    .rename(columns={"index": "Column"}),
        #         use_container_width=True
        #     )
        zero_counts = {}
        for col in numeric_df_raw.columns:
            zero_counts[col] = int((numeric_df_raw[col] == 0).sum())
        suggested_invalid_cols = []
        
        for col in numeric_df_raw.columns:
            zero_count = int((numeric_df_raw[col] == 0).sum())
            zero_counts[col] = zero_count
        
        # Hardcoded priority
            if col in invalid_zero_columns:
                suggested_invalid_cols.append(col)
        
        
        
        zero_series = pd.Series(zero_counts)
        zero_series = zero_series[zero_series > 0]
        
        if zero_series.empty:
            st.success("No suspicious zero values found!")
        else:
            st.dataframe(
                zero_series.rename("Zero Count")
                           .reset_index()
                           .rename(columns={"index": "Column"}),
                use_container_width=True
            )
    
    # ───────── MISSING VALUES ─────────
    with col_mv:
        st.markdown("**Missing Values**")
        temp_df = df.copy()

# convert selected zero columns to NaN temporarily
        for col in suggested_invalid_cols:
            temp_df[col] = temp_df[col].replace(0, np.nan)
        
        missing = temp_df.isnull().sum()
        missing = missing[missing > 0]
    
        if missing.empty:
            st.success("No missing values")
        else:
            st.dataframe(
                missing.rename("Missing Count")
                       .reset_index()
                       .rename(columns={"index": "Column"}),
                use_container_width=True
            )
    
    # ───────── OUTLIERS ─────────
    with col_out:
        st.markdown("**Outliers (IQR)**")
        if numeric_df_raw.empty:
            st.info("No numeric columns")
        else:
            outlier_counts = get_per_col_outliers(numeric_df_raw)
            outlier_counts = outlier_counts[outlier_counts > 0]
    
            if outlier_counts.empty:
                st.success("No outliers")
            else:
                st.dataframe(
                    outlier_counts.rename("Outlier Count")
                                  .reset_index()
                                  .rename(columns={"index": "Column"}),
                    use_container_width=True
                )
    







    # with col_mv:
    #     st.markdown("**Missing Values per Column**")
    #     missing = df.isnull().sum()
    #     missing = missing[missing > 0]
    #     if missing.empty:
    #         st.success("No missing values found!")
    #     else:
    #         st.dataframe(
    #             missing.rename("Missing Count")
    #                    .reset_index()
    #                    .rename(columns={"index": "Column"}),
    #             use_container_width=True
    #         )

    # with col_out:
    #     st.markdown("**Outliers per Numeric Column (IQR)**")
    #     if numeric_df_raw.empty:
    #         st.info("No numeric columns.")
    #     else:
    #         outlier_counts = get_per_col_outliers(numeric_df_raw)
    #         outlier_counts = outlier_counts[outlier_counts > 0]
    #         if outlier_counts.empty:
    #             st.success("No outliers detected!")
    #         else:
    #             st.dataframe(
    #                 outlier_counts.rename("Outlier Count")
    #                               .reset_index()
    #                               .rename(columns={"index": "Column"}),
    #                 use_container_width=True
    #             )

    st.markdown("---")
    st.subheader("Cleaning Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_option = st.selectbox(
            "Missing Value Strategy",
            ["None", "Drop Rows", "Mean", "Median"]
        )

    with col2:
        outlier_option = st.selectbox(
            "Outlier Handling",
            ["None", "Remove IQR"]
        )

    with col3:
        zero_option = st.selectbox(
            "Zero Handling",
            ["None", "Convert Zero → NaN (then fill)"]
        )
    st.markdown("### 🔧 Select columns where 0 is invalid")

    selected_zero_cols = st.multiselect(
        "Columns",
        numeric_df_raw.columns,
        default=suggested_invalid_cols
    )

    # ── APPLY CLEANING ──
    if st.button("✅ Apply Cleaning"):
        df_clean = df.copy()

        # # -------- ZERO HANDLING --------
        # if zero_option == "Convert Zero → NaN (then fill)":
        #     for col in df_clean.select_dtypes(include=np.number).columns:
        #         df_clean[col] = df_clean[col].replace(0, np.nan)
        if zero_option == "Convert Zero → NaN (then fill)":
            for col in selected_zero_cols:
                df_clean[col] = df_clean[col].replace(0, np.nan)

        # -------- MISSING VALUES --------
        if missing_option == "Drop Rows":
            df_clean = df_clean.dropna()

        elif missing_option == "Mean":
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

        elif missing_option == "Median":
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

        # -------- OUTLIER HANDLING --------
        if outlier_option == "Remove IQR":
            numeric_part = df_clean.select_dtypes(include=np.number)
            Q1 = numeric_part.quantile(0.25)
            Q3 = numeric_part.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_part < (Q1 - 1.5 * IQR)) | (numeric_part > (Q3 + 1.5 * IQR))).any(axis=1)
            df_clean = df_clean[mask]

        # -------- FINAL CLEAN DATA --------
        df_clean_num = df_clean.select_dtypes(include=np.number)
        st.session_state["df_clean"] = df_clean_num

        # ── AFTER CLEANING ──
        st.subheader("After Cleaning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", len(df))
        c2.metric("Cleaned Rows", len(df_clean_num))
        c3.metric("Remaining Missing", int(df_clean_num.isnull().sum().sum()))
        c4.metric("Features", len(df_clean_num.columns))

        st.dataframe(df_clean_num.head(), use_container_width=True)

        st.success("✅ Data cleaned successfully!")



# ═══════════════════════════ FEATURE SELECTION ═══════════════════════════
elif page == "Feature Selection":
    st.title("🎯 Feature Selection")

    info_box(
        "A dataset can have many columns, but not all of them are useful for prediction. "
        "Feature Selection identifies and keeps only the columns that matter most — "
        "this improves model accuracy, reduces training time, and cuts out noise."
    )

    df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

    if df_used.shape[1] < 2:
        st.error("Need at least 2 numeric columns. Please complete Data Cleaning first.")
        st.stop()

    # Step 1 — Target
    st.subheader("Step 1 — Choose Your Target Column")
    info_box(
        "The target column is what you want the model to learn to predict — for example, "
        "'Survived' in Titanic, 'Outcome' in a diabetes dataset, or 'Price' in a housing dataset. "
        "All other columns become inputs (features) the model uses to make that prediction."
    )
    target = st.selectbox("Select Target Column", df_used.columns)
    task = detect_task_type(df_used[target])

    task_icon = "🟢" if task == "classification" else "🔵"
    task_explain = (
        f"Since `{target}` has **{df_used[target].nunique()} unique values**, "
        "this is treated as a **Classification** problem — the model will predict a category/class."
        if task == "classification"
        else
        f"Since `{target}` has **{df_used[target].nunique()} unique values**, "
        "this is treated as a **Regression** problem — the model will predict a continuous number."
    )
    st.markdown(f"{task_icon} **Detected Task: {task.upper()}** — {task_explain}")

    X = df_used.drop(columns=[target])
    y = df_used[target]

    # Step 2 — Method
    st.subheader("Step 2 — Choose a Feature Importance Method")
    info_box(
        "**None** — All columns are kept as features with no ranking. \n\n"
        "**Correlation** — Measures the linear relationship between each feature and the target. "
        "Simple and fast — best for datasets where relationships are roughly linear."
    )
    method = st.selectbox("Feature Importance Method", ["None", "Correlation"])

    selected_features = list(X.columns)

    if method == "Correlation":
        corr = df_used.corr()[target].abs().drop(target).sort_values(ascending=True)
        fig = px.bar(corr, orientation="h", color=corr,
                     color_continuous_scale="Blues",
                     labels={"value": "Absolute Correlation", "index": "Feature"},
                     title="Absolute Correlation with Target (higher = more related)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Step 3 — How Many Top Features to Keep?")
        info_box(
            "Keep the N columns most correlated with the target. "
            "Columns with near-zero correlation contribute very little and can safely be dropped."
        )
        top_n = st.slider("Top N Features to Keep", 1, len(X.columns), min(5, len(X.columns)))
        selected_features = corr.sort_values(ascending=False).head(top_n).index.tolist()

    else:
        st.info("No method selected — all features will be passed to the model.")

    st.markdown("---")
    st.success(f"✅ Features selected for training ({len(selected_features)}): {selected_features}")
    st.session_state["selected_features"] = selected_features
    st.session_state["target"] = target
    st.session_state["task"] = task


# ═══════════════════════════ MODEL TRAINING ═══════════════════════════
elif page == "Model Training":
    st.title("🤖 Model Training")

    info_box(
        "This is where the model is built and rigorously evaluated. "
        "The pipeline: split data → scale features → train model → evaluate on held-out test data → "
        "validate stability with K-Fold cross-validation."
    )

    df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
    features = st.session_state.get("selected_features", list(df_used.columns[:-1]))
    target   = st.session_state.get("target", df_used.columns[-1])
    task     = st.session_state.get("task", detect_task_type(df_used[target]))

    # Configuration summary
    st.subheader("Current Pipeline Configuration")
    info_box(
        f"Task: <b>{task.upper()}</b> &nbsp;|&nbsp; "
        f"Target (what we predict): <b>{target}</b> &nbsp;|&nbsp; "
        f"Input features (what we use): <b>{len(features)} columns</b> → {features}"
    )

    X = df_used[features].fillna(0)
    y = df_used[target]

    if task == "classification":
        y, le = encode_target(y)
        model_options = ["Logistic Regression", "SVM", "KNN"]
    else:
        le = None
        model_options = ["Linear Regression", "SVM", "KNN"]

    st.markdown("---")
    st.subheader("Training Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider(
            "Test Size",
            0.1, 0.4, 0.2, step=0.05,
            help="Fraction of the data held back for testing. 0.2 = 80% train, 20% test."
        )
    with col2:
        k_folds = st.slider(
            "K-Fold Splits",
            2, 10, 5,
            help="Number of cross-validation folds. More folds = more reliable estimate but slower."
        )
    with col3:
        model_type = st.selectbox(
            "Model",
            model_options,
            help="Choose the algorithm that best fits your data patterns."
        )

    info_box(
        f"<b>Test Size {int(test_size*100)}%</b>: model trains on "
        f"{int((1-test_size)*len(X))} rows, tested on {int(test_size*len(X))} unseen rows. &nbsp;|&nbsp; "
        f"<b>K-Fold ({k_folds})</b>: the dataset is split into {k_folds} parts; "
        f"the model trains {k_folds} times, each time using a different part as test, "
        f"giving an averaged score that is more reliable than a single train/test split."
    )

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            X_all_s   = scaler.fit_transform(X)

            if task == "classification":
                if model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_type == "SVM":
                    model = SVC(probability=True)
                elif model_type == "KNN":
                    model = KNeighborsClassifier()
            else:
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "SVM":
                    model = SVR()
                elif model_type == "KNN":
                    model = KNeighborsRegressor()

            try:
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                st.markdown("---")
                st.subheader("📈 Performance Metrics")

                # ── CLASSIFICATION ──
                if task == "classification":
                    avg_strategy = "binary" if y.nunique() == 2 else "weighted"

                    acc  = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)
                    rec  = recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)
                    f1   = f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)

                    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    cv_res = cross_validate(
                        model, X_all_s, y, cv=cv,
                        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
                    )
                    avg_acc  = cv_res["test_accuracy"].mean()
                    avg_prec = cv_res["test_precision_weighted"].mean()
                    avg_rec  = cv_res["test_recall_weighted"].mean()
                    avg_f1   = cv_res["test_f1_weighted"].mean()

                    st.markdown("**Test Set Scores** *(single held-out test split)*")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{acc:.4f}")
                    c2.metric("Precision", f"{prec:.4f}")
                    c3.metric("Recall",    f"{rec:.4f}")
                    c4.metric("F1 Score",  f"{f1:.4f}")

                    info_box(
                        "<b>Accuracy</b> = % of all predictions that are correct. &nbsp;"
                        "<b>Precision</b> = of all positive predictions, how many were truly positive (avoids false alarms). &nbsp;"
                        "<b>Recall</b> = of all actual positives, how many did we correctly identify (avoids missed cases). &nbsp;"
                        "<b>F1 Score</b> = balance between Precision and Recall — most useful when classes are imbalanced."
                    )

                    st.markdown(f"**{k_folds}-Fold Cross-Validation Averages** *(more reliable, averaged across {k_folds} runs)*")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Avg Accuracy",  f"{avg_acc:.4f}")
                    k2.metric("Avg Precision", f"{avg_prec:.4f}")
                    k3.metric("Avg Recall",    f"{avg_rec:.4f}")
                    k4.metric("Avg F1 Score",  f"{avg_f1:.4f}")

                    # Grouped bar chart
                    metrics_df = pd.DataFrame({
                        "Metric":        ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Test Score":    [acc,      prec,      rec,    f1],
                        "CV Avg Score":  [avg_acc,  avg_prec,  avg_rec, avg_f1],
                    })
                    fig_bar = px.bar(
                        metrics_df.melt(id_vars="Metric", var_name="Evaluation", value_name="Score"),
                        x="Metric", y="Score", color="Evaluation", barmode="group",
                        color_discrete_sequence=["#FF4B4B", "#4a7cff"],
                        range_y=[0, 1],
                        title="Performance Metrics — Test Set vs K-Fold Average"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    info_box(
                        "Each cell shows how many samples were predicted as a given class vs their actual class. "
                        "Diagonal = correct predictions. Off-diagonal = mistakes."
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                                       labels={"x": "Predicted", "y": "Actual"},
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ── REGRESSION ──
                else:
                    r2   = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae  = mean_absolute_error(y_test, y_pred)

                    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    cv_res = cross_validate(
                        model, X_all_s, y, cv=cv,
                        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
                    )
                    avg_r2   = cv_res["test_r2"].mean()
                    avg_rmse = np.sqrt((-cv_res["test_neg_mean_squared_error"]).mean())
                    avg_mae  = (-cv_res["test_neg_mean_absolute_error"]).mean()

                    st.markdown("**Test Set Scores**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score", f"{r2:.4f}")
                    c2.metric("RMSE",     f"{rmse:.4f}")
                    c3.metric("MAE",      f"{mae:.4f}")

                    st.markdown(f"**{k_folds}-Fold Cross-Validation Averages**")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Avg R²",   f"{avg_r2:.4f}")
                    k2.metric("Avg RMSE", f"{avg_rmse:.4f}")
                    k3.metric("Avg MAE",  f"{avg_mae:.4f}")

                    info_box(
                        "<b>R²</b> = proportion of target variance explained by the model (1.0 = perfect). &nbsp;"
                        "<b>RMSE</b> = average prediction error in the target's units — penalises large errors more. &nbsp;"
                        "<b>MAE</b> = average absolute error — treats all errors equally regardless of size."
                    )

                    metrics_df = pd.DataFrame({
                        "Metric":       ["R²",    "RMSE",     "MAE"],
                        "Test Score":   [r2,      rmse,       mae],
                        "CV Avg Score": [avg_r2,  avg_rmse,   avg_mae],
                    })
                    fig_bar = px.bar(
                        metrics_df.melt(id_vars="Metric", var_name="Evaluation", value_name="Score"),
                        x="Metric", y="Score", color="Evaluation", barmode="group",
                        color_discrete_sequence=["#FF4B4B", "#4a7cff"],
                        title="Regression Metrics — Test Set vs K-Fold Average"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.success("✅ Training complete!")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.info("Make sure you have completed Data Cleaning and Feature Selection before training.")