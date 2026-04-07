import streamlit as st
import boto3
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Return Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 TSLA Cumulative Return Predictor")
st.markdown("Enter stock cumulative returns from the S&P 500 to predict **TSLA's future cumulative return**.")

# ─────────────────────────────────────────────
# AWS CONFIG (edit these values)
# ─────────────────────────────────────────────
ENDPOINT_NAME  = "tsla-kpca-lasso-endpoint-1"
BUCKET_NAME    = "connor-whitaker-s3-bucket"
EXPLAINER_KEY  = "explainer/explainer_pca.shap"
REGION         = "us-east-1"   # change if your region differs

# ─────────────────────────────────────────────
# SIDEBAR — AWS Credentials
# ─────────────────────────────────────────────
st.sidebar.header("🔑 AWS Credentials")
st.sidebar.markdown("Paste your temporary SageMaker credentials below.")

access_key    = st.sidebar.text_input("Access Key ID",    type="password")
secret_key    = st.sidebar.text_input("Secret Access Key", type="password")
session_token = st.sidebar.text_input("Session Token",     type="password")

credentials_provided = all([access_key, secret_key, session_token])

# ─────────────────────────────────────────────
# BOTO3 SESSION HELPER
# ─────────────────────────────────────────────
@st.cache_resource
def get_clients(access_key, secret_key, session_token, region):
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name=region
    )
    sm_runtime = session.client('sagemaker-runtime')
    s3_client  = session.client('s3')
    return sm_runtime, s3_client


# ─────────────────────────────────────────────
# LOAD SP500 COLUMN NAMES (for input form)
# ─────────────────────────────────────────────
@st.cache_data
def get_feature_columns():
    """
    Returns the expected feature column names (S&P500 stocks excluding TSLA).
    Update this list to match your actual X column names from the notebook.
    """
    # These are the _CR_Cum column names from your X dataframe
    # Replace with the actual columns from your dataset if different
    sample_columns = [
        'MSFT_CR_Cum', 'AAPL_CR_Cum', 'AMZN_CR_Cum', 'GOOGL_CR_Cum',
        'NVDA_CR_Cum', 'META_CR_Cum', 'BRK-B_CR_Cum', 'JPM_CR_Cum',
        'JNJ_CR_Cum', 'V_CR_Cum'
    ]
    return sample_columns


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Make a Prediction", "📊 SHAP Explainability", "ℹ️ About"])

# ── TAB 1: PREDICTION ──────────────────────────
with tab1:
    st.subheader("Enter Feature Values")
    st.markdown(
        "Enter the **cumulative returns** for the S&P 500 stocks below. "
        "These correspond to the `_CR_Cum` features used to train the model."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upload a CSV of feature values**")
        uploaded_file = st.file_uploader(
            "Upload a CSV where each row is one observation and columns match training features",
            type=["csv"]
        )

    with col2:
        st.markdown("**Or enter values manually (single observation)**")
        feature_cols = get_feature_columns()
        manual_values = {}
        for col_name in feature_cols:
            manual_values[col_name] = st.number_input(
                col_name, value=1.0, format="%.4f", key=col_name
            )

    st.divider()

    predict_btn = st.button("🚀 Predict", type="primary", disabled=not credentials_provided)

    if not credentials_provided:
        st.info("👈 Enter your AWS credentials in the sidebar to enable predictions.")

    if predict_btn and credentials_provided:
        try:
            sm_runtime, s3_client = get_clients(access_key, secret_key, session_token, REGION)

            # Prepare input data
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(input_df)} rows from CSV.")
            else:
                input_df = pd.DataFrame([manual_values])

            # Convert to JSON for endpoint
            payload = json.dumps(input_df.values.tolist())

            with st.spinner("Calling SageMaker endpoint..."):
                response = sm_runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='application/json',
                    Accept='application/json',
                    Body=payload
                )
                result = json.loads(response['Body'].read().decode())

            predictions = result.get('predictions', result)

            st.success("✅ Prediction complete!")
            st.metric(
                label="Predicted TSLA Cumulative Return",
                value=f"{predictions[0]:.4f}" if len(predictions) == 1 else f"{len(predictions)} predictions"
            )

            if len(predictions) > 1:
                pred_df = pd.DataFrame({'Predicted_TSLA_Cumulative_Return': predictions})
                st.dataframe(pred_df)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ── TAB 2: SHAP ────────────────────────────────
with tab2:
    st.subheader("SHAP Waterfall Plot — Local Explainability")
    st.markdown(
        "This shows how each KernelPCA component contributed to a single prediction."
    )

    shap_btn = st.button("📥 Load SHAP Explainer from S3", disabled=not credentials_provided)

    if shap_btn and credentials_provided:
        try:
            _, s3_client = get_clients(access_key, secret_key, session_token, REGION)

            with st.spinner("Downloading SHAP explainer from S3..."):
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=EXPLAINER_KEY)
                explainer_bytes = obj['Body'].read()

            with open('/tmp/explainer_pca.shap', 'wb') as f:
                f.write(explainer_bytes)

            with open('/tmp/explainer_pca.shap', 'rb') as f:
                explainer = shap.Explainer.load(f)

            st.success("✅ SHAP explainer loaded!")

            # If the user uploaded data, use it; otherwise show a placeholder message
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                shap_values = explainer(input_df.values[:1])
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            else:
                st.info("Upload a CSV in the Prediction tab and re-click to generate the SHAP plot for that observation.")

        except Exception as e:
            st.error(f"❌ Could not load SHAP explainer: {e}")

# ── TAB 3: ABOUT ───────────────────────────────
with tab3:
    st.subheader("About This App")
    st.markdown("""
    **HW5 — Dimensionality Reduction, Option 1**

    This app deploys a Scikit-learn pipeline trained to predict **TSLA's 5-day cumulative future return**
    using the cumulative returns of other S&P 500 stocks as features.

    ### Pipeline Architecture
    | Step | Method |
    |------|--------|
    | Missing Values | `SimpleImputer(strategy='median')` |
    | Scaling | `RobustScaler` |
    | Dimensionality Reduction | `KernelPCA(kernel='rbf', n_components=90% variance)` |
    | Regression | `Lasso` (tuned via GridSearchCV) |

    ### Model Tuning
    GridSearchCV was used with two varying parameters:
    - `kpca__gamma` — controls the RBF kernel bandwidth
    - `model__alpha` — controls Lasso regularization strength

    ### Deployment
    - **Model hosted on**: AWS SageMaker (SKLearn endpoint)
    - **Artifacts stored in**: `s3://connor-whitaker-s3-bucket/`
    - **Explainability**: SHAP LinearExplainer
    """)
