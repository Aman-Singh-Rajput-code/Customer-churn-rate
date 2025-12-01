'''import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/churn_model.pkl")  # update path as needed
# If using a scaler or encoder, load those too:
# scaler = joblib.load("model/scaler.pkl")
# encoder = joblib.load("model/encoder.pkl")

# Streamlit app UI
st.title("Customer Churn Prediction")

# User inputs
creditscore = st.number_input('Credit Score', min_value=0, step=1)
geography = st.text_input('Geography')
gender = st.text_input("Gender")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=1.0)
numofproducts = st.number_input("Number of Products", min_value=0, max_value=10, step=1)
hascrcard = st.number_input("Has Credit Card", min_value=0, max_value=1, step=1)
isactivemember = st.number_input("Is Active Member", min_value=0, max_value=1, step=1)
estimatedsalary = st.number_input("Estimated Salary", min_value=0.0, step=1.0)

if st.button('Predict'):
    # Convert input to dataframe
    input_data = pd.DataFrame([{
        'creditscore': creditscore,
        'geography': geography,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'numofproducts': numofproducts,
        'hascrcard': hascrcard,
        'isactivemember': isactivemember,
        'estimatedsalary': estimatedsalary
    }])

    # Preprocess if needed (example only)
    # input_data['gender'] = input_data['gender'].map({'Male': 0, 'Female': 1})
    # input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success('Prediction: Customer is **not** likely to leave the company')
    else:
        st.warning('Prediction: Customer is **likely** to leave the company')
'''
# Web/steanlit_app.py (patched)
# Robust Streamlit app for Customer Churn Prediction
# - Safe model loading (relative path, optional download via MODEL_DOWNLOAD_URL)
# - Lazy cached model loading to avoid import-time crashes
# - Friendly UI and error handling

'''
import os
import pathlib
import logging
import joblib
import requests
import streamlit as st
import pandas as pd

LOG = logging.getLogger(__name__)

# ------------------ Model loader config ------------------
HERE = pathlib.Path(__file__).resolve().parent
# repo layout: /<repo_root>/model/churn_model.pkl
MODEL_DIR = (HERE / ".." / "model").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = "churn_model.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def download_file(url: str, dest: pathlib.Path, chunk_size: int = 8192):
    """Stream-download a file to dest (overwrites if exists)."""
    LOG.info("Downloading model from %s to %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    LOG.info("Download finished: %s", dest)


def _load_model_from_disk(path: pathlib.Path):
    LOG.info("Attempting to load model from %s", path)
    return joblib.load(path)


def ensure_model_available():
    """Ensure model exists locally: load if present or download if env var provided."""
    # 1) If file exists already, load it
    if MODEL_PATH.exists():
        return _load_model_from_disk(MODEL_PATH)

    # 2) Try to download if MODEL_DOWNLOAD_URL env var is set
    download_url = os.environ.get("MODEL_DOWNLOAD_URL")
    if download_url:
        try:
            LOG.info("Model not found locally. Downloading from MODEL_DOWNLOAD_URL...")
            download_file(download_url, MODEL_PATH)
            LOG.info("Download complete, attempting load...")
            return _load_model_from_disk(MODEL_PATH)
        except Exception as e:
            LOG.exception("Failed downloading or loading model from %s: %s", download_url, e)

    # 3) Nothing worked — show helpful UI message and stop the app
    st.error(
        "Model file not found. Expected:\n\n"
        f"  {MODEL_PATH}\n\n"
        "Options to fix:\n"
        "  • Add the file to your repo under `model/churn_model.pkl` and push.\n"
        "  • Or host the file (S3/GCS/GitHub release) and set the MODEL_DOWNLOAD_URL env var\n"
        "    to a direct download URL in the Streamlit app settings."
    )
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


# Lazy cached loader so the model is only loaded once per process
@st.cache_resource
def get_model():
    return ensure_model_available()


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer data and click **Predict** to determine churn likelihood.")

# Input widgets (kept similar to original but with types enforced)
creditscore = st.number_input('Credit Score', min_value=0, step=1, value=600)
geography = st.text_input('Geography', value='France')
gender = st.text_input('Gender', value='Male')
age = st.number_input('Age', min_value=0, max_value=120, step=1, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, step=1, value=1)
balance = st.number_input('Balance', min_value=0.0, step=1.0, value=0.0, format="%f")
numofproducts = st.number_input('Number of Products', min_value=0, max_value=10, step=1, value=1)
hascrcard = st.number_input('Has Credit Card (0 or 1)', min_value=0, max_value=1, step=1, value=1)
isactivemember = st.number_input('Is Active Member (0 or 1)', min_value=0, max_value=1, step=1, value=1)
estimatedsalary = st.number_input('Estimated Salary', min_value=0.0, step=1.0, value=50000.0, format="%f")

# Build input DataFrame in the format the model expects
input_data = pd.DataFrame([
    {
        'creditscore': int(creditscore),
        'geography': str(geography),
        'gender': str(gender),
        'age': int(age),
        'tenure': int(tenure),
        'balance': float(balance),
        'numofproducts': int(numofproducts),
        'hascrcard': int(hascrcard),
        'isactivemember': int(isactivemember),
        'estimatedsalary': float(estimatedsalary)
    }
])

st.write("---")
st.subheader("Input preview")
st.dataframe(input_data)

# Prediction button
if st.button('Predict'):
    try:
        # Load model (cached)
        try:
            model = get_model()
        except FileNotFoundError:
            st.stop()

        # Make prediction
        prediction = model.predict(input_data)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            # If binary, take second column as prob of class 1
            prob_of_churn = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        else:
            prob_of_churn = None

        # Interpret prediction: assume 1 -> churn, 0 -> no churn
        pred_label = int(prediction[0])
        if pred_label == 1:
            st.warning('Prediction: Customer is **likely** to leave the company (churn).')
        else:
            st.success('Prediction: Customer is **not** likely to leave the company (no churn).')

        if prob_of_churn is not None:
            st.info(f"Model confidence (P(churn)=): {prob_of_churn:.3f}")

    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        st.error(f"Prediction failed: {e}")


# Optional: show troubleshooting/helpful hints
with st.expander("Deployment / troubleshooting tips"):
    st.markdown(
        "- If you see `Model file not found`, add `model/churn_model.pkl` to your repo or set `MODEL_DOWNLOAD_URL` env var.\n"
        "- For large models, consider hosting on S3 and using `MODEL_DOWNLOAD_URL`.\n"
        "- If you changed feature names during training, make sure the input column names above exactly match those used when training the model."
    )
'''

# Web/steanlit_app.py
# Clean Streamlit app for Customer Churn Prediction
# - set_page_config is the very first Streamlit call
# - robust model loader (can download via MODEL_DOWNLOAD_URL)
# - safe lazy loading with st.cache_resource

import os
import pathlib
import logging
import joblib
import requests
import streamlit as st
import pandas as pd

# ------------------ IMPORTANT: set page config first ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
# ----------------------------------------------------------------------

LOG = logging.getLogger(__name__)

# ------------------ Model loader config ------------------
HERE = pathlib.Path(__file__).resolve().parent
# If churn_model.pkl is in repo root (you uploaded it there), this path points to it.
MODEL_FILENAME = "churn_model.pkl"
MODEL_PATH = (HERE / ".." / MODEL_FILENAME).resolve()

# Ensure parent dir exists (safe no-op)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: pathlib.Path, chunk_size: int = 8192):
    """Stream-download a file to dest (overwrites if exists)."""
    LOG.info("Downloading model from %s to %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    LOG.info("Download finished: %s", dest)


def _load_model_from_disk(path: pathlib.Path):
    LOG.info("Attempting to load model from %s", path)
    return joblib.load(path)


def ensure_model_available():
    """Ensure model exists locally: load if present or download if env var provided."""
    # 1) If file exists already, load it
    if MODEL_PATH.exists():
        return _load_model_from_disk(MODEL_PATH)

    # 2) Try to download if MODEL_DOWNLOAD_URL env var is set
    download_url = os.environ.get("MODEL_DOWNLOAD_URL")
    if download_url:
        try:
            LOG.info("Model not found locally. Downloading from MODEL_DOWNLOAD_URL...")
            download_file(download_url, MODEL_PATH)
            LOG.info("Download complete, attempting load...")
            return _load_model_from_disk(MODEL_PATH)
        except Exception as e:
            LOG.exception("Failed downloading or loading model from %s: %s", download_url, e)

    # 3) Nothing worked — show helpful UI message and stop the app
    st.error(
        "Model file not found. Expected:\n\n"
        f"  {MODEL_PATH}\n\n"
        "Options to fix:\n"
        "  • Add the file to your repo root with the name `churn_model.pkl` and push.\n"
        "  • Or host the file (S3/GCS/GitHub release) and set the MODEL_DOWNLOAD_URL env var\n"
        "    to a direct download URL in the Streamlit app settings."
    )
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


# Now safe to use Streamlit's caching decorator (page config already set)
@st.cache_resource
def get_model():
    return ensure_model_available()


# ------------------ Streamlit UI ------------------
st.title("Customer Churn Prediction")
st.write("Enter customer data and click **Predict** to determine churn likelihood.")

# Input widgets
creditscore = st.number_input('Credit Score', min_value=0, step=1, value=600)
geography = st.text_input('Geography', value='France')
gender = st.text_input('Gender', value='Male')
age = st.number_input('Age', min_value=0, max_value=120, step=1, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, step=1, value=1)
balance = st.number_input('Balance', min_value=0.0, step=1.0, value=0.0, format="%f")
numofproducts = st.number_input('Number of Products', min_value=0, max_value=10, step=1, value=1)
hascrcard = st.number_input('Has Credit Card (0 or 1)', min_value=0, max_value=1, step=1, value=1)
isactivemember = st.number_input('Is Active Member (0 or 1)', min_value=0, max_value=1, step=1, value=1)
estimatedsalary = st.number_input('Estimated Salary', min_value=0.0, step=1.0, value=50000.0, format="%f")

# Build input DataFrame in the format the model expects
input_data = pd.DataFrame([{
    'creditscore': int(creditscore),
    'geography': str(geography),
    'gender': str(gender),
    'age': int(age),
    'tenure': int(tenure),
    'balance': float(balance),
    'numofproducts': int(numofproducts),
    'hascrcard': int(hascrcard),
    'isactivemember': int(isactivemember),
    'estimatedsalary': float(estimatedsalary)
}])

st.write("---")
st.subheader("Input preview")
st.dataframe(input_data)

# Prediction button
if st.button('Predict'):
    try:
        # Load model (cached)
        try:
            model = get_model()
        except FileNotFoundError:
            # ensure_model_available already showed a st.error; stop execution
            st.stop()

        # Make prediction
        prediction = model.predict(input_data)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            # If binary, take second column as prob of class 1
            prob_of_churn = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        else:
            prob_of_churn = None

        # Interpret prediction: assume 1 -> churn, 0 -> no churn
        pred_label = int(prediction[0])
        if pred_label == 1:
            st.warning('Prediction: Customer is **likely** to leave the company (churn).')
        else:
            st.success('Prediction: Customer is **not** likely to leave the company (no churn).')

        if prob_of_churn is not None:
            st.info(f"Model confidence (P(churn)=): {prob_of_churn:.3f}")

    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        st.error(f"Prediction failed: {e}")

# Optional: show troubleshooting/helpful hints
with st.expander("Deployment / troubleshooting tips"):
    st.markdown(
        "- If you see `Model file not found`, add `churn_model.pkl` to your repo root or set `MODEL_DOWNLOAD_URL` env var.\n"
        "- For large models, consider hosting on S3 and using `MODEL_DOWNLOAD_URL`.\n"
        "- If you changed feature names during training, make sure the input column names above exactly match those used when training the model."
    )
