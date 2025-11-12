# =========================================================
# TechnoHacks Internship – End-to-End Data Science Dashboard
# Mentor: Sandip Gavit
# Author: <Your Name>
# Run: streamlit run app.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import io
import json
from datetime import datetime

# ---------------------------------------------
# Streamlit page setup
# ---------------------------------------------
st.set_page_config(
    page_title="TechnoHacks Data Science Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------
# App State
# ---------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_feat" not in st.session_state:
    st.session_state.df_feat = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "y_proba" not in st.session_state:
    st.session_state.y_proba = None

# ---------------------------------------------
# Caching helpers
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_titanic() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data(show_spinner=False)
def save_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------------------------
# Header
# ---------------------------------------------
st.title("📚 TechnoHacks Internship – All Tasks (1–12) Dashboard")
st.caption("Mentor: Sandip Gavit | Built with Streamlit | Dataset: Titanic (default) or upload your own")

# ---------------------------------------------
# Sidebar – Navigation & Global Controls
# ---------------------------------------------
st.sidebar.header("🧭 Navigation")
task = st.sidebar.radio(
    "Go to Task:",
    [
        "Task 1: Data Collection",
        "Task 2: Data Cleaning",
        "Task 3: EDA",
        "Task 4: Data Visualization",
        "Task 5: Feature Engineering",
        "Task 6: Statistical Analysis",
        "Task 7: ML Model Development",
        "Task 8: Model Evaluation",
        "Task 9: NLP",
        "Task 10: Data Science Dashboard",
        "Task 11: Case Study",
        "Task 12: Capstone Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Dataset Source")
data_source = st.sidebar.selectbox(
    "Choose dataset source",
    ["Use default Titanic", "Upload CSV"]
)

if data_source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        try:
            st.session_state.df_raw = pd.read_csv(file)
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
else:
    # default
    if st.session_state.df_raw is None:
        st.session_state.df_raw = load_default_titanic()

# Quick peek button
if st.sidebar.button("🔍 Peek Raw Data (Top 10)"):
    st.sidebar.success("Showing top 10 rows in main area")
    st.write("🔍 **Raw Data Preview (Top 10 Rows)**")
    st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

# ---------------------------------------------
# Utility functions
# ---------------------------------------------
def clean_data(df: pd.DataFrame, numeric_strategy: str = "mean", drop_threshold: float = 0.5):
    """
    Clean data:
    - Drop columns with too many missing values (above threshold).
    - Impute numeric with mean/median/zero.
    - Impute non-numeric with 'Unknown'.
    - Standardize numeric columns.
    Returns cleaned df and fitted scaler.
    """
    df = df.copy()

    # Drop columns with > threshold missing
    miss_ratio = df.isna().mean()
    to_drop = miss_ratio[miss_ratio > drop_threshold].index.tolist()
    if len(to_drop) > 0:
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    # Numeric & categorical splits
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # Impute numeric
    if numeric_strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif numeric_strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif numeric_strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # Impute categorical
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Standardize numeric
    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler

def engineer_features(df: pd.DataFrame):
    """
    Example feature engineering for Titanic-like schema:
    - FamilySize = SibSp + Parch
    - Title extracted from Name (if present)
    - Encode Sex, Embarked
    """
    df = df.copy()

    # FamilySize
    if set(["SibSp", "Parch"]).issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0)
    else:
        df["FamilySize"] = 0

    # Extract Title from Name if available
    if "Name" in df.columns:
        df["Title"] = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False)
        df["Title"] = df["Title"].fillna("Unknown")
    else:
        df["Title"] = "Unknown"

    # Encode Sex & Embarked if present
    enc = LabelEncoder()
    for col in ["Sex", "Embarked", "Title"]:
        if col in df.columns:
            df[col] = enc.fit_transform(df[col].astype(str))

    return df

def split_train_model(df: pd.DataFrame, target_col: str, model_name: str, test_size: float, random_state: int, **params):
    """
    Split, train & return model and datasets.
    model_name: 'RandomForest', 'LogisticRegression', 'GradientBoosting'
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in dataframe columns!")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
            solver="lbfgs"
        )
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=random_state
        )
    else:
        raise ValueError("Unsupported model!")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    return model, X_train, X_test, y_train, y_test, y_pred, y_proba

def plot_confusion_matrix(cm: np.ndarray, labels=None, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc_pr(y_test, y_proba):
    # ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax1.plot([0,1],[0,1], linestyle="--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        st.pyplot(fig1)

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        st.pyplot(fig2)
    else:
        st.info("Model doesn't provide probabilities; ROC/PR not available.")

def downloadable_button(data: bytes, label: str, file_name: str, mime: str):
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        use_container_width=True
    )

# ---------------------------------------------
# TASK 1: Data Collection
# ---------------------------------------------
if task == "Task 1: Data Collection":
    st.header("📥 Task 1: Data Collection")
    st.write("""
    **Goal:** Load a public dataset or uploaded CSV and save it locally as a structured file.  
    - Using Titanic dataset by default (from GitHub).  
    - You can upload your own CSV in the sidebar.  
    - Click the download button to save a local copy.
    """)
    df = st.session_state.df_raw
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Shape:** "
                f"`{df.shape[0]} rows × {df.shape[1]} columns`")

    st.subheader("Download Raw Dataset")
    raw_csv = save_csv_bytes(df)
    downloadable_button(raw_csv, "⬇️ Download raw_dataset.csv", f"raw_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ---------------------------------------------
# TASK 2: Data Cleaning
# ---------------------------------------------
elif task == "Task 2: Data Cleaning":
    st.header("🧹 Task 2: Data Cleaning")
    df = st.session_state.df_raw.copy()

    st.info("Adjust cleaning rules on the left, then click **Clean Data**.")
    st.sidebar.subheader("Cleaning Options")
    numeric_strategy = st.sidebar.radio("Numeric Imputation Strategy", ["mean", "median", "zero"], index=0)
    drop_threshold = st.sidebar.slider("Drop columns above missing ratio", 0.0, 1.0, 0.5, 0.05)

    if st.button("✨ Clean Data"):
        df_clean, scaler = clean_data(df, numeric_strategy=numeric_strategy, drop_threshold=drop_threshold)
        st.session_state.df_clean = df_clean
        st.success("Data cleaned successfully!")
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean.head(20), use_container_width=True)
        st.markdown(f"**Shape after cleaning:** `{df_clean.shape[0]} rows × {df_clean.shape[1]} columns`")

        st.subheader("Missing Value Heatmap (After Cleaning)")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(df_clean.isna(), cbar=False, yticklabels=False, ax=ax)
        st.pyplot(fig)

        st.subheader("Download Cleaned Dataset")
        cleaned_csv = save_csv_bytes(df_clean)
        downloadable_button(cleaned_csv, "⬇️ Download cleaned_dataset.csv", f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.warning("Click **Clean Data** to process the dataset.")

# ---------------------------------------------
# TASK 3: EDA
# ---------------------------------------------
elif task == "Task 3: EDA":
    st.header("🔎 Task 3: Exploratory Data Analysis")
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw.copy()

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        num_col = st.selectbox("Select numeric column", numeric_cols, index=0)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=num_col, nbins=30, title=f"Histogram: {num_col}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[num_col], ax=ax2)
            ax2.set_title(f"Boxplot: {num_col}")
            st.pyplot(fig2)
    else:
        st.info("No numeric columns found.")

    st.subheader("Correlation Heatmap")
    if numeric_cols:
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False, ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("No numeric columns for correlation.")

# ---------------------------------------------
# TASK 4: Data Visualization
# ---------------------------------------------
elif task == "Task 4: Data Visualization":
    st.header("📊 Task 4: Data Visualization")
    df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw.copy()

    st.write("Create interactive charts with Plotly.")

    cols = df.columns.tolist()
    x_col = st.selectbox("X-axis", cols, index=0)
    y_col = st.selectbox("Y-axis", cols, index=min(1, len(cols)-1))
    color_col = st.selectbox("Color (optional)", ["(none)"] + cols, index=0)

    if st.button("📈 Make Charts"):
        color_arg = None if color_col == "(none)" else color_col

        st.subheader("Bar / Histogram (auto)")
        if pd.api.types.is_numeric_dtype(df[y_col]):
            fig_bar = px.bar(df, x=x_col, y=y_col, color=color_arg, title=f"Bar: {y_col} by {x_col}")
        else:
            fig_bar = px.histogram(df, x=x_col, color=color_arg, title=f"Histogram: {x_col}")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Scatter (if numeric)")
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            fig_sc = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=f"Scatter: {y_col} vs {x_col}")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Scatter needs both X and Y numeric.")

        st.subheader("Box Plot")
        try:
            fig_box = px.box(df, x=x_col, y=y_col, color=color_arg, title=f"Box: {y_col} by {x_col}")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.warning(f"Box plot not available for the selected combination. {e}")

# ---------------------------------------------
# TASK 5: Feature Engineering
# ---------------------------------------------
elif task == "Task 5: Feature Engineering":
    st.header("⚙️ Task 5: Feature Engineering")
    base_df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw.copy()

    st.write("Applies Titanic-inspired engineered features (FamilySize, Title) + encodes Sex/Embarked.")
    if st.button("🧩 Apply Feature Engineering"):
        df_feat = engineer_features(base_df)
        st.session_state.df_feat = df_feat
        st.success("Feature engineering applied.")
        st.dataframe(df_feat.head(20), use_container_width=True)
        st.markdown(f"**New shape:** `{df_feat.shape[0]} rows × {df_feat.shape[1]} columns`")

        st.subheader("Feature Correlation (numeric only)")
        num_cols = df_feat.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(df_feat[num_cols].corr(), cmap="viridis", annot=False, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns.")

        st.subheader("Download Engineered Dataset")
        downloadable_button(save_csv_bytes(df_feat), "⬇️ Download engineered_dataset.csv", f"engineered_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.info("Click **Apply Feature Engineering** to proceed.")

# ---------------------------------------------
# TASK 6: Statistical Analysis
# ---------------------------------------------
elif task == "Task 6: Statistical Analysis":
    st.header("📐 Task 6: Statistical Analysis")
    df = (st.session_state.df_feat or st.session_state.df_clean) if st.session_state.df_feat is not None else st.session_state.df_raw.copy()

    st.write("Perform correlation and hypothesis testing (t-tests) between two groups.")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        target_like = st.selectbox("Binary Column (0/1) for Group Split", options=num_cols, index=0)
        test_col = st.selectbox("Numeric Column for t-test", options=[c for c in num_cols if c != target_like], index=0)

        # Split groups by threshold at median if not binary
        if df[target_like].nunique() == 2:
            group1 = df[df[target_like] == sorted(df[target_like].unique())[0]][test_col].dropna()
            group2 = df[df[target_like] == sorted(df[target_like].unique())[1]][test_col].dropna()
        else:
            thresh = df[target_like].median()
            group1 = df[df[target_like] <= thresh][test_col].dropna()
            group2 = df[df[target_like] > thresh][test_col].dropna()

        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        st.metric("T-statistic", f"{t_stat:.4f}")
        st.metric("P-value", f"{p_val:.6f}")

        st.subheader("Distribution by Groups")
        fig = go.Figure()
        fig.add_trace(go.Box(y=group1, name="Group A"))
        fig.add_trace(go.Box(y=group2, name="Group B"))
        fig.update_layout(title=f"Boxplot of {test_col} by groups based on {target_like}")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# TASK 7: Machine Learning Model Development
# ---------------------------------------------
elif task == "Task 7: ML Model Development":
    st.header("🤖 Task 7: Machine Learning Model Development")
    df_base = st.session_state.df_feat or st.session_state.df_clean or st.session_state.df_raw
    df = df_base.copy()

    st.info("Select target column and model, then train.")
    target_col = st.selectbox("Target Column", options=df.columns.tolist(), index=df.columns.tolist().index("Survived") if "Survived" in df.columns else 0)

    model_name = st.selectbox("Model", ["RandomForest", "LogisticRegression", "GradientBoosting"], index=0)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)

    st.subheader("Hyperparameters")
    params = {}
    if model_name == "RandomForest":
        params["n_estimators"] = st.slider("n_estimators", 50, 1000, 300, 50)
        params["max_depth"] = st.slider("max_depth (None = 0)", 0, 30, 0, 1)
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, 1)
        if params["max_depth"] == 0:
            params["max_depth"] = None
    elif model_name == "LogisticRegression":
        params["C"] = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
        params["max_iter"] = st.slider("max_iter", 100, 5000, 1000, 100)
    elif model_name == "GradientBoosting":
        params["n_estimators"] = st.slider("n_estimators", 50, 1000, 300, 50)
        params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
        params["max_depth"] = st.slider("max_depth", 1, 10, 3, 1)

    if st.button("🚀 Train Model"):
        try:
            model, X_train, X_test, y_train, y_test, y_pred, y_proba = split_train_model(
                df, target_col, model_name, test_size, random_state, **params
            )
            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_proba = y_proba

            st.success(f"{model_name} trained successfully.")
            st.write("**X_train shape:**", X_train.shape, "| **X_test shape:**", X_test.shape)

            # Feature importance (if available)
            st.subheader("Feature Importance / Coefficients")
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                st.bar_chart(importances)
            elif hasattr(model, "coef_"):
                coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values(ascending=False)
                st.bar_chart(coefs)
            else:
                st.info("This model does not expose feature importances.")

            # Save model as JSON-like (just parameters)
            try:
                params_json = json.dumps(model.get_params(), indent=2)
                st.download_button("⬇️ Download Model Params (JSON)", params_json, file_name=f"{model_name}_params.json")
            except Exception:
                pass

        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------------------------------------------
# TASK 8: Model Evaluation
# ---------------------------------------------
elif task == "Task 8: Model Evaluation":
    st.header("📏 Task 8: Model Evaluation")
    if st.session_state.model is None or st.session_state.y_pred is None:
        st.warning("No trained model found. Train in Task 7 first.")
    else:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_proba = st.session_state.y_proba

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Class 0", "Class 1"] if len(np.unique(y_test)) == 2 else None
        plot_confusion_matrix(cm, labels=labels)

        st.subheader("ROC & PR Curves")
        plot_roc_pr(y_test, y_proba)

        # Download predictions
        st.subheader("Download Predictions")
        pred_df = pd.DataFrame({
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred).reset_index(drop=True)
        })
        if y_proba is not None:
            pred_df["proba"] = pd.Series(y_proba).reset_index(drop=True)
        downloadable_button(save_csv_bytes(pred_df), "⬇️ Download predictions.csv", f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ---------------------------------------------
# TASK 9: NLP
# ---------------------------------------------
elif task == "Task 9: NLP":
    st.header("💬 Task 9: Natural Language Processing (Sentiment)")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

    st.write("Enter text or upload a .txt file to analyze sentiment.")
    text_input = st.text_area("Type text here", height=120, placeholder="I loved the service! But delivery was slow...")
    file_txt = st.file_uploader("Or upload a .txt file", type=["txt"])

    texts = []
    if text_input.strip():
        texts.append(text_input.strip())
    if file_txt is not None:
        try:
            texts.append(file_txt.read().decode("utf-8"))
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    if st.button("🧠 Analyze Sentiment"):
        if not texts:
            st.warning("Please provide some text.")
        else:
            results = []
            for t in texts:
                # Split long text by lines for more granularity
                for line in [ln.strip() for ln in t.splitlines() if ln.strip()]:
                    scores = sia.polarity_scores(line)
                    results.append({"text": line, **scores})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)

            # Distribution of compound score
            st.subheader("Compound Sentiment Distribution")
            fig = px.histogram(res_df, x="compound", nbins=40, title="Sentiment (compound) distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            downloadable_button(save_csv_bytes(res_df), "⬇️ Download sentiment_results.csv", f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ---------------------------------------------
# TASK 10: Data Science Dashboard
# ---------------------------------------------
elif task == "Task 10: Data Science Dashboard":
    st.header("🖥️ Task 10: Dashboard Demonstration")
    st.write("""
    This entire Streamlit app **is** your Data Science dashboard:
    - Data ingestion, cleaning, EDA, visualization  
    - Feature engineering, ML training & evaluation  
    - NLP and downloads  
    Use this for your **video demo** on LinkedIn/YouTube and tag **TechnoHacks** & **Mentor Sandip Gavit**.
    """)

    st.subheader("Mini KPI Cards (from cleaned/engineered data)")
    df = st.session_state.df_feat or st.session_state.df_clean or st.session_state.df_raw
    n_rows, n_cols = df.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{n_cols:,}")
    c3.metric("Missing (%)", f"{df.isna().mean().mean()*100:.2f}%")

    st.subheader("Quick Visual – Numeric Column Distribution")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        col = st.selectbox("Pick a numeric column", num_cols)
        st.bar_chart(df[col].value_counts().head(25))
    else:
        st.info("No numeric columns available in current dataset.")

# ---------------------------------------------
# TASK 11: Data Science Case Study
# ---------------------------------------------
elif task == "Task 11: Case Study":
    st.header("📚 Task 11: Case Study (Titanic Survival Example)")
    st.write("""
    **Problem**: Predict passenger survival on the Titanic.  
    **Approach**:  
    1) Collected data (public GitHub).  
    2) Cleaned missing values; standardized numeric features.  
    3) EDA with distributions & correlations.  
    4) Feature engineering (FamilySize, Title, encoding).  
    5) Trained ML models (RF/LR/GB).  
    6) Evaluated with Accuracy, Precision, Recall, F1; ROC/PR curves.  
    **Outcome**: Working end-to-end pipeline with downloadable artifacts.
    """)

# ---------------------------------------------
# TASK 12: Capstone Project
# ---------------------------------------------
elif task == "Task 12: Capstone Project":
    st.header("🏆 Task 12: Capstone Project – End-to-End Pipeline")
    st.write("""
    This tab finalizes your capstone by combining:
    - Data Collection → Cleaning → EDA → Visualization  
    - Feature Engineering → ML → Evaluation → NLP → Dashboard  
    **Action**: Export your cleaned/engineered datasets and model params as deliverables.
    """)

    df_clean = st.session_state.df_clean or st.session_state.df_raw
    df_feat = st.session_state.df_feat
    model = st.session_state.model

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Export Cleaned Dataset")
        downloadable_button(save_csv_bytes(df_clean), "⬇️ Download cleaned_dataset.csv", f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

        if df_feat is not None:
            st.subheader("Export Engineered Dataset")
            downloadable_button(save_csv_bytes(df_feat), "⬇️ Download engineered_dataset.csv", f"engineered_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        else:
            st.info("Engineer features in Task 5 to enable this.")

    with colB:
        st.subheader("Export Trained Model Params")
        if model is not None:
            try:
                params_json = json.dumps(model.get_params(), indent=2)
                st.download_button("⬇️ Download Model Params (JSON)", params_json, file_name=f"model_params_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
            except Exception:
                st.warning("Model parameters could not be serialized.")
        else:
            st.info("Train a model in Task 7 to enable this.")

    st.success("Capstone ready! Record your walkthrough video and tag **TechnoHacks** & **Mentor Sandip Gavit**.")
