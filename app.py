"""
PipeDash — Interactive ML Pipeline Dashboard
Uses the Movie dataset to walk through a complete ML workflow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PipeDash - ML Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 1.5rem; }

/* ---------- step-bar ---------- */
.step-bar {
    display: flex; justify-content: center; gap: 0;
    margin-bottom: 2rem; padding: 0 1rem;
    flex-wrap: wrap;
}
.step-item { display: flex; align-items: center; gap: 0; }
.step-circle {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .85rem; flex-shrink: 0;
}
.step-circle.done   { background: #111; color: #fff; }
.step-circle.active  { background: #555; color: #fff; }
.step-circle.todo   { background: #E5E5E5; color: #999; }
.step-line { width: 38px; height: 3px; border-radius: 2px; }
.step-line.done { background: #111; }
.step-line.todo { background: #E5E5E5; }
.step-label {
    font-size: .6rem; text-align: center; color: #888;
    max-width: 60px; margin-top: 4px; line-height: 1.15;
}
.step-wrapper { display: flex; flex-direction: column; align-items: center; }

/* ---------- cards ---------- */
.glass-card {
    background: #fff;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* ---------- section title ---------- */
.section-title {
    font-size: 1.35rem; font-weight: 700;
    color: #111;
    margin-bottom: .6rem;
}
.section-sub {
    font-size: .85rem; color: #888; margin-bottom: 1rem;
}

/* ---------- metric cards ---------- */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #FAFAFA;
    border: 1px solid #E5E5E5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 800; color: #111; }
.metric-label { font-size: .75rem; color: #888; margin-top: .2rem; }

/* ---------- hero ---------- */
.hero { text-align: center; margin-bottom: .5rem; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color: #111; }
.hero p { color: #888; font-size: 1rem; margin-top: -0.5rem; }

/* ---------- buttons ---------- */
div.stButton > button {
    background: #111; color: #fff;
    border: none; border-radius: 8px;
    padding: .55rem 2rem; font-weight: 600;
}
div.stButton > button:hover {
    background: #333;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────
STEP_NAMES = [
    "Problem",
    "Data",
    "EDA",
    "Clean",
    "Features",
    "Split",
    "Model",
    "Train",
    "Tune",
]
TOTAL_STEPS = len(STEP_NAMES)

PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#7C3AED"
ACCENT2 = "#F59E0B"


def init_state():
    """Initialise session-state keys."""
    defaults = dict(
        current_step=1,
        problem_type=None,
        df=None,
        df_clean=None,
        target=None,
        selected_features=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        trained_models={},
        split_ratio=0.8,
        k_folds=5,
        model_choices=[],
        final_features=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def advance(step: int):
    """Set the current step to at least *step*."""
    st.session_state.current_step = max(st.session_state.current_step, step)


def render_step_bar():
    """Draw the horizontal step-progress indicator."""
    cur = st.session_state.current_step
    items_html = ""
    for i, name in enumerate(STEP_NAMES, 1):
        cls = "done" if i < cur else ("active" if i == cur else "todo")
        items_html += f"""
        <div class="step-item">
            <div class="step-wrapper">
                <div class="step-circle {cls}">{i}</div>
                <div class="step-label">{name}</div>
            </div>
        """
        if i < TOTAL_STEPS:
            lcls = "done" if i < cur else "todo"
            items_html += f'<div class="step-line {lcls}"></div>'
        items_html += "</div>"
    st.markdown(f'<div class="step-bar">{items_html}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True
        )


def metric_cards(pairs: list[tuple[str, str]]):
    """Render a row of metric cards. pairs = [(value, label), ...]"""
    cards = "".join(
        f'<div class="metric-card"><div class="metric-value">{v}</div>'
        f'<div class="metric-label">{l}</div></div>'
        for v, l in pairs
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    init_state()

    # Hero
    st.markdown(
        '<div class="hero"><h1>PipeDash</h1>'
        "<p>Interactive Machine-Learning Pipeline Dashboard</p></div>",
        unsafe_allow_html=True,
    )
    render_step_bar()

    # ── STEP 1 ─ Problem Type ──────────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("① Problem Type", "Choose the type of ML problem you want to solve.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Classification", width="stretch", key="btn_cls"):
                st.session_state.problem_type = "Classification"
                advance(2)
        with col2:
            if st.button("Regression", width="stretch", key="btn_reg"):
                st.session_state.problem_type = "Regression"
                advance(2)
        if st.session_state.problem_type:
            st.success(f"Selected: **{st.session_state.problem_type}**")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 2:
        return

    # ── STEP 2 ─ Data Input ────────────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "② Data Input & PCA Visualisation",
            "Load the Movie dataset, select your target, and explore data shape in PCA space.",
        )

        @st.cache_data
        def load_data():
            return pd.read_csv("assets/merged_movie_data.csv")

        df = load_data().copy()
        # Drop only strictly identifier/URL columns, keep titles for context
        id_cols = ["id", "movie_id", "homepage"]
        for c in id_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        st.session_state.df = df.copy()

        st.dataframe(df.head(10), width="stretch")
        metric_cards(
            [
                (str(df.shape[0]), "Rows"),
                (str(df.shape[1]), "Columns"),
                (str(df.select_dtypes(include="number").shape[1]), "Numeric"),
                (str(df.isnull().sum().sum()), "Missing"),
            ]
        )

        target = st.selectbox(
            "Select target column",
            df.columns.tolist(),
            index=df.columns.tolist().index("vote_average")
            if "vote_average" in df.columns
            else 0,
            key="sel_target",
        )
        st.session_state.target = target

        feature_cols = df.select_dtypes(include="number").columns.tolist()
        if target in feature_cols:
            feature_cols.remove(target)

        selected = st.multiselect(
            "Select features for PCA (numeric only recommended)",
            feature_cols,
            default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols,
            key="sel_features_pca",
        )
        if not selected:
            selected = feature_cols

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected].fillna(0))
        n_comp = min(3, len(selected))

        if n_comp >= 2:
            pca = PCA(n_components=n_comp)
            pcs = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(n_comp)])
            pca_df["target"] = df[target].astype(str)

            tab2d, tab3d = st.tabs(
                ["2-D PCA", "3-D PCA"] if n_comp >= 3 else ["2-D PCA", "—"]
            )
            with tab2d:
                fig2 = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color="target",
                    template=PLOTLY_TEMPLATE,
                    title="PCA — 2-D Projection",
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                )
                fig2.update_layout(margin=dict(t=40, b=30), height=420)
                st.plotly_chart(fig2, width="stretch")
            if n_comp >= 3:
                with tab3d:
                    fig3 = px.scatter_3d(
                        pca_df,
                        x="PC1",
                        y="PC2",
                        z="PC3",
                        color="target",
                        template=PLOTLY_TEMPLATE,
                        title="PCA — 3-D Projection",
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                    )
                    fig3.update_layout(margin=dict(t=40, b=10), height=500)
                    st.plotly_chart(fig3, width="stretch")

            # Explained variance
            exp_var = pca.explained_variance_ratio_
            fig_var = px.bar(
                x=[f"PC{i + 1}" for i in range(n_comp)],
                y=exp_var,
                labels={"x": "Component", "y": "Explained Variance Ratio"},
                template=PLOTLY_TEMPLATE,
                title="Explained Variance by Component",
                color_discrete_sequence=[ACCENT],
            )
            fig_var.update_layout(margin=dict(t=40, b=30), height=300)
            st.plotly_chart(fig_var, width="stretch")

        if st.button("Continue to EDA", key="btn_to_eda"):
            advance(3)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 3:
        return

    # ── STEP 3 ─ EDA ───────────────────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "③ Exploratory Data Analysis",
            "Descriptive statistics, distributions, and correlations.",
        )
        df = st.session_state.df.copy()
        target = st.session_state.target

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.style.format("{:.2f}"), width="stretch")

        # Distribution histograms
        st.subheader("Feature Distributions")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(num_cols):
                    c = num_cols[idx]
                    fig = px.histogram(
                        df,
                        x=c,
                        nbins=30,
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[ACCENT],
                        title=c,
                    )
                    fig.update_layout(
                        margin=dict(t=35, b=20, l=20, r=20),
                        height=240,
                        showlegend=False,
                    )
                    col.plotly_chart(fig, width="stretch")

        # Box plots
        st.subheader("Box Plots")
        for i in range(0, len(num_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(num_cols):
                    c = num_cols[idx]
                    fig = px.box(
                        df,
                        y=c,
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[ACCENT2],
                        title=c,
                    )
                    fig.update_layout(
                        margin=dict(t=35, b=20, l=20, r=20),
                        height=240,
                        showlegend=False,
                    )
                    col.plotly_chart(fig, width="stretch")

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="Plasma",
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(margin=dict(t=40, b=20), height=550)
        st.plotly_chart(fig_corr, width="stretch")

        if st.button("Continue to Data Cleaning", key="btn_to_clean"):
            advance(4)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 4:
        return

    # ── STEP 4 ─ Data Engineering & Cleaning ───
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "④ Data Engineering & Cleaning",
            "Handle missing values and detect / remove outliers.",
        )
        df = st.session_state.df.copy()
        target = st.session_state.target
        num_cols = df.select_dtypes(include="number").columns.tolist()

        # --- Missing value imputation ---
        st.subheader("Missing Value Imputation")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.info("No missing values detected.")
        else:
            st.dataframe(miss.rename("Missing Count"), width="stretch")
            imp_method = st.selectbox(
                "Imputation method", ["mean", "median", "mode"], key="imp_method"
            )
            if st.button("Apply Imputation", key="btn_imp"):
                for c in miss.index:
                    is_num = pd.api.types.is_numeric_dtype(df[c])
                    if imp_method == "mean" and is_num:
                        df[c] = df[c].fillna(df[c].mean())
                    elif imp_method == "median" and is_num:
                        df[c] = df[c].fillna(df[c].median())
                    else:
                        # Mode works for both numeric and categorical
                        if not df[c].mode().empty:
                            df[c] = df[c].fillna(df[c].mode()[0])
                st.session_state.df = df.copy()
                st.success("Imputation applied!")

        # --- Outlier detection ---
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox(
            "Detection method",
            ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"],
            key="outlier_method",
        )

        feature_for_outlier = [c for c in num_cols if c != target]
        outlier_mask = pd.Series(False, index=df.index)

        if outlier_method == "IQR":
            for c in feature_for_outlier:
                Q1 = df[c].quantile(0.25)
                Q3 = df[c].quantile(0.75)
                iqr = Q3 - Q1
                lower = Q1 - 1.5 * iqr
                upper = Q3 + 1.5 * iqr
                outlier_mask |= (df[c] < lower) | (df[c] > upper)
        elif outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(df[feature_for_outlier].fillna(0))
            outlier_mask = pd.Series(preds == -1, index=df.index)
        elif outlier_method == "DBSCAN":
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[feature_for_outlier].fillna(0))
            db = DBSCAN(eps=2.0, min_samples=5)
            labels = db.fit_predict(scaled)
            outlier_mask = pd.Series(labels == -1, index=df.index)
        elif outlier_method == "OPTICS":
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[feature_for_outlier].fillna(0))
            opt = OPTICS(min_samples=5)
            labels = opt.fit_predict(scaled)
            outlier_mask = pd.Series(labels == -1, index=df.index)

        n_outliers = outlier_mask.sum()
        metric_cards(
            [
                (str(n_outliers), "Outliers Detected"),
                (f"{n_outliers / len(df) * 100:.1f}%", "of Data"),
                (str(len(df) - n_outliers), "Clean Rows"),
            ]
        )

        if n_outliers > 0:
            with st.expander(f"View {n_outliers} outlier rows"):
                st.dataframe(df[outlier_mask], width="stretch")

            if st.button("Remove Outliers", key="btn_rm_outliers"):
                df = df[~outlier_mask].reset_index(drop=True)
                st.session_state.df = df.copy()
                st.success(f"Removed {n_outliers} outlier rows. New shape: {df.shape}")

        st.session_state.df_clean = df.copy()

        if st.button("Continue to Feature Selection", key="btn_to_fs"):
            advance(5)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 5:
        return

    # ── STEP 5 ─ Feature Selection ─────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "⑤ Feature Selection",
            "Rank features by variance, correlation, and information gain.",
        )
        df = (
            st.session_state.df_clean
            if st.session_state.df_clean is not None
            else st.session_state.df.copy()
        )
        target = st.session_state.target
        feature_cols = df.select_dtypes(include="number").columns.tolist()
        if target in feature_cols:
            feature_cols.remove(target)
        X = df[feature_cols]
        y = df[target]

        tab_vt, tab_cor, tab_ig = st.tabs(
            ["Variance Threshold", "Correlation", "Information Gain"]
        )

        # Variance threshold
        with tab_vt:
            threshold = st.slider(
                "Variance threshold", 0.0, 2.0, 0.1, 0.05, key="vt_slider"
            )
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X.fillna(0))
            variances = pd.Series(selector.variances_, index=feature_cols).sort_values(
                ascending=False
            )
            kept = variances[variances >= threshold].index.tolist()

            fig_vt = px.bar(
                x=variances.index,
                y=variances.values,
                labels={"x": "Feature", "y": "Variance"},
                template=PLOTLY_TEMPLATE,
                title="Feature Variances",
                color=variances.values,
                color_continuous_scale="Plasma",
            )
            fig_vt.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=ACCENT2,
                annotation_text="Threshold",
            )
            fig_vt.update_layout(margin=dict(t=40, b=30), height=350)
            st.plotly_chart(fig_vt, width="stretch")
            st.caption(
                f"Features above threshold: **{len(kept)}** / {len(feature_cols)}"
            )

        # Correlation with target
        with tab_cor:
            corr_with_target = (
                X.fillna(0).corrwith(y).abs().sort_values(ascending=False)
            )
            fig_ct = px.bar(
                x=corr_with_target.index,
                y=corr_with_target.values,
                labels={"x": "Feature", "y": "|Correlation with target|"},
                template=PLOTLY_TEMPLATE,
                title="Absolute Correlation with Target",
                color=corr_with_target.values,
                color_continuous_scale="Viridis",
            )
            fig_ct.update_layout(margin=dict(t=40, b=30), height=350)
            st.plotly_chart(fig_ct, width="stretch")

        # Information Gain
        with tab_ig:
            mi_func = (
                mutual_info_classif
                if st.session_state.problem_type == "Classification"
                else mutual_info_regression
            )
            mi_scores = mi_func(X.fillna(0), y, random_state=42)
            mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(
                ascending=False
            )
            fig_mi = px.bar(
                x=mi_series.index,
                y=mi_series.values,
                labels={"x": "Feature", "y": "Mutual Information"},
                template=PLOTLY_TEMPLATE,
                title="Information Gain (Mutual Information)",
                color=mi_series.values,
                color_continuous_scale="Magma",
            )
            fig_mi.update_layout(margin=dict(t=40, b=30), height=350)
            st.plotly_chart(fig_mi, width="stretch")

        # Let user pick final features
        st.subheader("Select Final Features")
        final_features = st.multiselect(
            "Pick features to use for modelling",
            feature_cols,
            default=feature_cols,
            key="final_feat_select",
        )
        st.session_state.final_features = final_features

        if st.button("Continue to Data Split", key="btn_to_split"):
            advance(6)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 6:
        return

    # ── STEP 6 ─ Data Split ────────────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("⑥ Data Split", "Split into training and testing sets.")

        df = (
            st.session_state.df_clean
            if st.session_state.df_clean is not None
            else st.session_state.df.copy()
        )
        target = st.session_state.target
        features = st.session_state.final_features or [
            c for c in df.columns if c != target
        ]
        X = df[features]
        y = df[target]

        ratio = st.slider(
            "Training set ratio", 0.5, 0.95, 0.8, 0.05, key="split_slider"
        )
        st.session_state.split_ratio = ratio

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ratio, random_state=42
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        metric_cards(
            [
                (str(X_train.shape[0]), "Train samples"),
                (str(X_test.shape[0]), "Test samples"),
                (f"{ratio * 100:.0f}%", "Train ratio"),
                (str(len(features)), "Features"),
            ]
        )

        # Pie chart
        fig_pie = px.pie(
            values=[X_train.shape[0], X_test.shape[0]],
            names=["Train", "Test"],
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[ACCENT, ACCENT2],
            title="Train / Test Split",
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=40, b=20), height=320)
        st.plotly_chart(fig_pie, width="stretch")

        if st.button("Continue to Model Selection", key="btn_to_model"):
            advance(7)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 7:
        return

    # ── STEP 7 ─ Model Selection ───────────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("⑦ Model Selection", "Choose one or more models to train.")

        is_cls = st.session_state.problem_type == "Classification"

        model_options = {
            "Logistic Regression" if is_cls else "Linear Regression": "lr",
            "SVM": "svm",
            "Random Forest": "rf",
            "K-Means (unsupervised)": "kmeans",
        }

        picks = st.multiselect(
            "Select models",
            list(model_options.keys()),
            default=[list(model_options.keys())[0]],
            key="model_picks",
        )
        st.session_state.model_choices = [model_options[p] for p in picks]

        svm_kernel = None
        if "svm" in st.session_state.model_choices:
            svm_kernel = st.selectbox(
                "SVM Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kern"
            )
            st.session_state["svm_kernel"] = svm_kernel

        if st.button("Continue to Training", key="btn_to_train"):
            advance(8)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 8:
        return

    # ── STEP 8 ─ Training & K-Fold Validation ──
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "⑧ Model Training & K-Fold Validation",
            "Train models, evaluate with K-Fold cross-validation, and check for overfitting / underfitting.",
        )

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        is_cls = st.session_state.problem_type == "Classification"

        k = st.slider("Number of K-Folds", 2, 20, 5, key="k_slider")
        st.session_state.k_folds = k

        if st.button("Train Models", key="btn_train"):
            models = {}
            svm_kernel = st.session_state.get("svm_kernel", "rbf")

            for mc in st.session_state.model_choices:
                if mc == "lr":
                    models["Linear/Logistic Regression"] = (
                        LogisticRegression(max_iter=1000, random_state=42)
                        if is_cls
                        else LinearRegression()
                    )
                elif mc == "svm":
                    models[f"SVM ({svm_kernel})"] = (
                        SVC(kernel=svm_kernel, random_state=42)
                        if is_cls
                        else SVR(kernel=svm_kernel)
                    )
                elif mc == "rf":
                    models["Random Forest"] = (
                        RandomForestClassifier(n_estimators=100, random_state=42)
                        if is_cls
                        else RandomForestRegressor(n_estimators=100, random_state=42)
                    )
                elif mc == "kmeans":
                    models["K-Means"] = None  # handled separately

            results = []
            trained_models_dict = {}

            for name, model in models.items():
                if model is None:
                    continue
                model.fit(X_train, y_train)
                trained_models_dict[name] = model

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                scoring = "accuracy" if is_cls else "r2"
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=KFold(n_splits=k, shuffle=True, random_state=42),
                    scoring=scoring,
                )

                if is_cls:
                    train_score = accuracy_score(y_train, train_pred)
                    test_score = accuracy_score(y_test, test_pred)
                    f1 = f1_score(y_test, test_pred, average="weighted")
                else:
                    train_score = r2_score(y_train, train_pred)
                    test_score = r2_score(y_test, test_pred)
                    f1 = mean_squared_error(y_test, test_pred)

                # Overfitting / underfitting
                gap = train_score - test_score
                if gap > 0.1:
                    fit_status = "OVERFITTING"
                elif test_score < 0.5:
                    fit_status = "UNDERFITTING"
                else:
                    fit_status = "Good Fit"

                results.append(
                    {
                        "Model": name,
                        "Train Score": round(train_score, 4),
                        "Test Score": round(test_score, 4),
                        "CV Mean": round(cv_scores.mean(), 4),
                        "CV Std": round(cv_scores.std(), 4),
                        "F1 / MSE": round(f1, 4),
                        "Fit Status": fit_status,
                    }
                )

            # K-Means (unsupervised)
            if "kmeans" in st.session_state.model_choices:
                n_clusters = len(y_train.unique()) if is_cls else 3
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                km.fit(X_train)
                trained_models_dict["K-Means"] = km
                results.append(
                    {
                        "Model": "K-Means",
                        "Train Score": "N/A",
                        "Test Score": "N/A",
                        "CV Mean": "N/A",
                        "CV Std": "N/A",
                        "F1 / MSE": f"Inertia: {km.inertia_:.0f}",
                        "Fit Status": "—",
                    }
                )

            st.session_state.trained_models = trained_models_dict

            res_df = pd.DataFrame(results)
            st.dataframe(res_df, width="stretch", hide_index=True)

            # Bar chart of scores
            plot_df = res_df[res_df["Train Score"] != "N/A"].copy()
            if not plot_df.empty:
                plot_df["Train Score"] = plot_df["Train Score"].astype(float)
                plot_df["Test Score"] = plot_df["Test Score"].astype(float)
                plot_df["CV Mean"] = plot_df["CV Mean"].astype(float)

                fig_scores = go.Figure()
                fig_scores.add_trace(
                    go.Bar(
                        name="Train",
                        x=plot_df["Model"],
                        y=plot_df["Train Score"],
                        marker_color=ACCENT,
                    )
                )
                fig_scores.add_trace(
                    go.Bar(
                        name="Test",
                        x=plot_df["Model"],
                        y=plot_df["Test Score"],
                        marker_color=ACCENT2,
                    )
                )
                fig_scores.add_trace(
                    go.Bar(
                        name="CV Mean",
                        x=plot_df["Model"],
                        y=plot_df["CV Mean"],
                        marker_color="#06B6D4",
                    )
                )
                fig_scores.update_layout(
                    barmode="group",
                    template=PLOTLY_TEMPLATE,
                    title="Model Performance Comparison",
                    yaxis_title="Score",
                    margin=dict(t=40, b=30),
                    height=380,
                )
                st.plotly_chart(fig_scores, width="stretch")

            # K-Fold per-fold detail for each model
            with st.expander("Per-Fold K-Fold Scores"):
                for name, model in trained_models_dict.items():
                    if name == "K-Means":
                        continue
                    scoring = "accuracy" if is_cls else "r2"
                    fold_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=KFold(n_splits=k, shuffle=True, random_state=42),
                        scoring=scoring,
                    )
                    fig_fold = px.bar(
                        x=[f"Fold {i + 1}" for i in range(k)],
                        y=fold_scores,
                        template=PLOTLY_TEMPLATE,
                        title=f"{name} — K-Fold Scores",
                        color_discrete_sequence=[ACCENT],
                        labels={"x": "Fold", "y": scoring.capitalize()},
                    )
                    fig_fold.add_hline(
                        y=fold_scores.mean(),
                        line_dash="dash",
                        line_color=ACCENT2,
                        annotation_text=f"Mean: {fold_scores.mean():.4f}",
                    )
                    fig_fold.update_layout(margin=dict(t=40, b=20), height=300)
                    st.plotly_chart(fig_fold, width="stretch")

        if st.session_state.trained_models:
            if st.button("Continue to Hyperparameter Tuning", key="btn_to_tune"):
                advance(9)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_step < 9:
        return

    # ── STEP 9 ─ Hyperparameter Tuning ─────────
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "⑨ Hyperparameter Tuning",
            "Fine-tune your models with GridSearch or RandomizedSearch.",
        )

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        is_cls = st.session_state.problem_type == "Classification"

        search_type = st.radio(
            "Search strategy",
            ["GridSearchCV", "RandomizedSearchCV"],
            horizontal=True,
            key="search_type",
        )

        # Define param grids for each model type
        param_grids = {}
        svm_kernel = st.session_state.get("svm_kernel", "rbf")
        for name in st.session_state.trained_models:
            if "Regression" in name or "Logistic" in name:
                if is_cls:
                    param_grids[name] = {
                        "C": [0.01, 0.1, 1, 10],
                        "max_iter": [500, 1000, 2000],
                    }
                else:
                    param_grids[name] = {"fit_intercept": [True, False]}
            elif "SVM" in name:
                param_grids[name] = {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear", "poly"],
                }
                if is_cls:
                    param_grids[name]["gamma"] = ["scale", "auto"]
            elif "Random Forest" in name:
                param_grids[name] = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                }

        model_to_tune = st.selectbox(
            "Select model to tune",
            [n for n in st.session_state.trained_models if n != "K-Means"],
            key="tune_model_select",
        )

        if model_to_tune and model_to_tune in param_grids:
            st.write("**Parameter grid:**")
            st.json(param_grids[model_to_tune])

            if st.button("Run Tuning", key="btn_tune"):
                base_model = st.session_state.trained_models[model_to_tune]

                # Score before tuning
                scoring = "accuracy" if is_cls else "r2"
                before_score = cross_val_score(
                    base_model,
                    X_train,
                    y_train,
                    cv=KFold(
                        n_splits=st.session_state.k_folds, shuffle=True, random_state=42
                    ),
                    scoring=scoring,
                ).mean()

                with st.spinner("Tuning in progress…"):
                    if search_type == "GridSearchCV":
                        searcher = GridSearchCV(
                            base_model,
                            param_grids[model_to_tune],
                            cv=KFold(
                                n_splits=st.session_state.k_folds,
                                shuffle=True,
                                random_state=42,
                            ),
                            scoring=scoring,
                            n_jobs=-1,
                        )
                    else:
                        searcher = RandomizedSearchCV(
                            base_model,
                            param_grids[model_to_tune],
                            cv=KFold(
                                n_splits=st.session_state.k_folds,
                                shuffle=True,
                                random_state=42,
                            ),
                            scoring=scoring,
                            n_jobs=-1,
                            n_iter=min(20, 999),
                            random_state=42,
                        )
                    searcher.fit(X_train, y_train)

                best = searcher.best_estimator_
                after_score = searcher.best_score_
                delta = after_score - before_score

                st.success(f"Best params: `{searcher.best_params_}`")

                metric_cards(
                    [
                        (f"{before_score:.4f}", "Before Tuning"),
                        (f"{after_score:.4f}", "After Tuning"),
                        (f"{'+' if delta >= 0 else ''}{delta:.4f}", "Delta"),
                    ]
                )

                # Before / After bar chart
                fig_ba = go.Figure()
                fig_ba.add_trace(
                    go.Bar(
                        name="Before",
                        x=[model_to_tune],
                        y=[before_score],
                        marker_color="#64748B",
                    )
                )
                fig_ba.add_trace(
                    go.Bar(
                        name="After",
                        x=[model_to_tune],
                        y=[after_score],
                        marker_color=ACCENT,
                    )
                )
                fig_ba.update_layout(
                    barmode="group",
                    template=PLOTLY_TEMPLATE,
                    title="Tuning Impact",
                    yaxis_title="Score",
                    margin=dict(t=40, b=30),
                    height=350,
                )
                st.plotly_chart(fig_ba, width="stretch")

                # CV results table
                cv_res = pd.DataFrame(searcher.cv_results_)
                cols_show = [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score",
                ]
                cols_show = [c for c in cols_show if c in cv_res.columns]
                st.dataframe(
                    cv_res[cols_show].sort_values("rank_test_score").head(10),
                    width="stretch",
                    hide_index=True,
                )

                # Update trained models with best estimator
                st.session_state.trained_models[model_to_tune] = best

                # Final test performance
                final_pred = best.predict(X_test)
                if is_cls:
                    final_acc = accuracy_score(y_test, final_pred)
                    final_f1 = f1_score(y_test, final_pred, average="weighted")
                    metric_cards(
                        [
                            (f"{final_acc:.4f}", "Final Test Accuracy"),
                            (f"{final_f1:.4f}", "Final Test F1"),
                        ]
                    )
                else:
                    final_r2 = r2_score(y_test, final_pred)
                    final_mse = mean_squared_error(y_test, final_pred)
                    metric_cards(
                        [
                            (f"{final_r2:.4f}", "Final Test R²"),
                            (f"{final_mse:.4f}", "Final Test MSE"),
                        ]
                    )
        elif model_to_tune:
            st.info("No tunable hyperparameters defined for this model.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div style="text-align:center;margin-top:2rem;color:#999;font-size:.8rem;">'
        "Built with Streamlit -- PipeDash v1.0"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
