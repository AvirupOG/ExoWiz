# app.py — Streamlit app for KOI classifier + Explainable AI assets
# ─────────────────────────────────────────────────────────────────────────────
# Features
# • Upload candidates CSV → score with your saved scikit-learn Pipeline (joblib bundle)
# • Global explanations: display permutation importance, SHAP beeswarm/bar, PDP images
# • Local explanations: browse top-k SHAP contributors per row (precomputed in Colab)
# • Case-insensitive feature matching, helpful validation, CSV download of predictions
#
# Expected repository layout (example):
#   artifacts/
#     koi_planet_classifier.joblib        # saved by your Colab training
#     roc_curve.png
#     pr_curve.png
#     feature_importance_permutation.png  # from explain_model()
#     shap_beeswarm.png
#     shap_bar.png
#     local_explanations_topk.csv
#     pdp_<feature>.png ...
#   app.py
#   requirements.txt  (see bottom comment)
#
# Run locally:   pip install -r requirements.txt && streamlit run app.py
# Deploy (Streamlit Cloud): point to this repo and app.py; set env vars if needed.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io, os, hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json, os
from skops.io import load as skload, get_untrusted_types
import base64

# ------------------------------
# Config & page
# ------------------------------

APP_TITLE = os.getenv("APP_TITLE", "ExoWiz")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/koi_planet_classifier.joblib")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", os.path.dirname(MODEL_PATH) or "artifacts")
MAX_PREVIEW_ROWS = int(os.getenv("MAX_PREVIEW_ROWS", "20"))

def get_base64_of_bin_file(bin_file):
    """Read a binary file and return base64-encoded string."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """Inject CSS with a base64 background image into the Streamlit app."""
    try:
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image not found: {png_file}")

st.set_page_config(page_title=APP_TITLE, layout="wide")
set_background(r"D:\New folder\Streamlit_App\web_bg.webp")
st.markdown(
    f"<h1 style='text-align: center; color: white;'>{APP_TITLE}</h1>",
    unsafe_allow_html=True
)
st.caption("Upload a CSV with the same feature columns used at training. The app will compute probabilities and show precomputed explainability assets.")

# ------------------------------
# Utilities
# ------------------------------

def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# --- sklearn compatibility shim (must be before skops.load) ---
try:
    import sklearn.compose._column_transformer as _ct
    # Some builds don’t expose this internal alias; it’s effectively a Python list
    if not hasattr(_ct, "_RemainderColsList"):
        _ct._RemainderColsList = list  # provide alias so skops can construct it
except Exception:
    pass
# ----------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_bundle(path_model="artifacts/model.skops",
                path_meta="artifacts/metadata.json"):
    if not os.path.exists(path_model):
        st.error(f"Missing model file: {path_model}")
        st.stop()
    if not os.path.exists(path_meta):
        st.error(f"Missing metadata file: {path_meta}")
        st.stop()

    # Expand the allow-list to match your HGB pipeline
    trusted_types = [
        # Core pipeline bits
        "sklearn.pipeline.Pipeline",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.impute._base.SimpleImputer",
        "sklearn.preprocessing._data.StandardScaler",

        # HistGradientBoosting internals
        "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
        "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",

        # Loss/link classes used by HGB
        "sklearn._loss.loss.HalfBinomialLoss",
        "sklearn._loss.link.LogitLink",
        "sklearn._loss.link.Interval",

        # NumPy dtypes used in persisted state
        "numpy.dtype",
    ]

    model = skload(path_model, trusted=trusted_types)

    with open(path_meta, "r") as f:
        meta = json.load(f)

    features = meta["features"]
    bundle = {"model": model, "features": features, "config": meta.get("config", {})}
    return bundle


def normalize_columns_case_insensitive(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    lower_map = {c.lower(): c for c in df.columns}
    required_lower = [c.lower() for c in feature_list]
    missing = [orig for orig in feature_list if orig.lower() not in lower_map]
    present_cols = [lower_map[c] for c in required_lower if c in lower_map]
    aligned = df[present_cols].copy()
    # rename to canonical names the model expects
    rename_map = {lower_map[c]: feat for c, feat in zip(required_lower, feature_list) if c in lower_map}
    aligned = aligned.rename(columns=rename_map)
    return aligned, missing


@st.cache_data(show_spinner=False)
def score_csv(file_bytes: bytes, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    bundle = load_bundle()
    model = bundle["model"]
    feature_list: List[str] = bundle["features"]

    df = pd.read_csv(io.BytesIO(file_bytes))
    aligned, missing = normalize_columns_case_insensitive(df, feature_list)
    if missing:
        return pd.DataFrame(), missing

    # Coerce to numeric; imputer in pipeline will handle NaN
    X = aligned.apply(pd.to_numeric, errors="coerce")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # fallback if using decision_function
        from scipy.special import expit
        scores = model.decision_function(X)
        proba = expit(scores)
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["planet_probability"] = proba
    out["prediction"] = np.where(pred == 1, "planet", "non-planet")
    out["threshold"] = threshold
    return out, []

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    st.markdown("**Model path**")
    st.code(MODEL_PATH)
    st.markdown("**Artifacts dir**")
    st.code(ARTIFACTS_DIR)

# ------------------------------
# Tabs
# ------------------------------

tab_pred, tab_global, tab_local = st.tabs(["🔮 Predictions", "📊 Global Explanations", "🔍 Local Explanations"]) 

# ------------------------------
# Predictions tab
# ------------------------------
with tab_pred:
    st.subheader("Run predictions on uploaded CSV")
    uploaded = st.file_uploader("Upload candidates CSV", type=["csv"], accept_multiple_files=False)

    if uploaded is None:
        st.info("Upload a CSV to begin. Columns are matched case-insensitively to the model's feature list.")
    else:
        # Preview
        try:
            preview_df = pd.read_csv(uploaded, nrows=min(MAX_PREVIEW_ROWS, 200))
            st.dataframe(preview_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        with st.spinner("Scoring…"):
            content = uploaded.getvalue()
            scored_df, missing = score_csv(content, threshold=thresh)

        if missing:
            st.error("Missing required feature columns: " + ", ".join(missing))
        else:
            st.success(f"Scored {len(scored_df)} rows.")

            # Results table
            st.dataframe(scored_df, use_container_width=True)

            # Probability histogram
            try:
                fig = plt.figure()
                plt.hist(scored_df["planet_probability"].values, bins=30)
                plt.xlabel("planet_probability"); plt.ylabel("count"); plt.title("Probability distribution")
                st.pyplot(fig, use_container_width=True)
            except Exception:
                pass

            # Download
            csv_buf = io.StringIO(); scored_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download predictions CSV",
                data=csv_buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
            )

# ------------------------------
# Global explanations tab
# ------------------------------
with tab_global:
    st.subheader("Global explainability assets (generated in Colab)")

    def _maybe_show(path: str, caption: str):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
            return True
        return False

    shown_any = False
    shown_any |= _maybe_show(os.path.join(ARTIFACTS_DIR, "feature_importance_permutation.png"), "Permutation Importance (Δ ROC-AUC)")
    shown_any |= _maybe_show(os.path.join(ARTIFACTS_DIR, "shap_beeswarm.png"), "SHAP Beeswarm")
    shown_any |= _maybe_show(os.path.join(ARTIFACTS_DIR, "shap_bar.png"), "SHAP Feature Importance (bar)")

    # PDP series
    pdp_imgs = sorted([f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("pdp_") and f.endswith(".png")]) if os.path.isdir(ARTIFACTS_DIR) else []
    if pdp_imgs:
        st.markdown("### Partial Dependence (top features)")
        cols = st.columns(2)
        for i, fname in enumerate(pdp_imgs):
            with cols[i % 2]:
                st.image(os.path.join(ARTIFACTS_DIR, fname), caption=fname, use_container_width=True)
        shown_any = True

    # Training metrics if present
    col1, col2 = st.columns(2)
    with col1:
        _maybe_show(os.path.join(ARTIFACTS_DIR, "roc_curve.png"), "Training ROC curve")
    with col2:
        _maybe_show(os.path.join(ARTIFACTS_DIR, "pr_curve.png"), "Training PR curve")

    if not shown_any:
        st.info("No explainability images found. In Colab, run your explainability step to generate them into the artifacts directory.")

# ------------------------------
# Local explanations tab
# ------------------------------
with tab_local:
    st.subheader("Local explanations (top-k SHAP contributors per row)")
    local_csv = os.path.join(ARTIFACTS_DIR, "local_explanations_topk.csv")
    if os.path.exists(local_csv):
        local_df = pd.read_csv(local_csv)
        if local_df.empty:
            st.info("local_explanations_topk.csv is empty.")
        else:
            left, right = st.columns([2,3])
            with left:
                st.markdown("**Browse rows**")
                row_id = st.selectbox("Row index (from explained sample)", local_df["row_index"].unique().tolist())
                row_expl = local_df[local_df["row_index"] == row_id].iloc[0]
                st.write("Top contributing features:")
                items = []
                for i in range(1, 6):
                    fname = row_expl.get(f"feat{i}_name")
                    fshap = row_expl.get(f"feat{i}_shap")
                    fval  = row_expl.get(f"feat{i}_value")
                    if pd.isna(fname):
                        continue
                    items.append((fname, float(fshap) if pd.notna(fshap) else 0.0, fval))
                # Display as a small table
                disp = pd.DataFrame(items, columns=["feature", "shap_value", "feature_value"])
                st.dataframe(disp, use_container_width=True)
            with right:
                # Simple SHAP bar chart for the selected row
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    vals = [x[1] for x in items]
                    names = [x[0] for x in items]
                    plt.barh(names[::-1], vals[::-1])
                    plt.axvline(0, linestyle=":", linewidth=1)
                    plt.xlabel("SHAP contribution (log-odds approx)")
                    plt.title(f"Row {row_id} — top feature contributions")
                    st.pyplot(fig, use_container_width=True)
                except Exception:
                    pass

            st.markdown("---")
            st.caption("Note: These explanations were precomputed in Colab on a sample of test rows. To align with new predictions, regenerate them after retraining.")
    else:
        st.info("local_explanations_topk.csv not found in artifacts. Generate it in Colab with your explainability step and include it in deployment.")

# Footer
st.markdown("---")
st.caption("Model: scikit-learn Pipeline loaded from joblib. Features matched case-insensitively. Explainability assets are static files produced offline in Colab.")

# ─────────────────────────────────────────────────────────────────────────────
# Suggested requirements.txt (create as a separate file in your repo)
# streamlit
# pandas
# numpy
# matplotlib
# scikit-learn
# joblib
# ─────────────────────────────────────────────────────────────────────────────