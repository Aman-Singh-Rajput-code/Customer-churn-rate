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
# --- robust model loader for Streamlit apps (drop-in replacement) ---
import os
import pathlib
import logging
import joblib
import requests
import streamlit as st

LOG = logging.getLogger(__name__)

# Resolve model directory relative to this script (avoids cwd problems)
HERE = pathlib.Path(__file__).resolve().parent
# Adjust this if your repo places model directory differently.
# Current repo tree: Web/steanlit_app.py  <- we place model at repo_root/model
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
    # Raise to stop further execution (Streamlit will display the error above)
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Lazy cached loader so the model is only loaded once per session process
@st.cache_resource
def get_model():
    return ensure_model_available()

# Use the model later in your code as:
# try:
#     model = get_model()
# except FileNotFoundError:
#     st.stop()
# ---------------------------------------------------------------------
