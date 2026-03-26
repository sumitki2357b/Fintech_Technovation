from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import pandas as pd
import pickle

from model.features import build_feature_matrix
from model.explain import compute_shap_matrix, serialise_shap_rows

app = FastAPI()

# CORS (required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
with open("data/models/lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Fraud Model API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    df = pd.read_csv(path)

    # limit (optional)
    df = df.head(20000)

    X, _ = build_feature_matrix(df)

    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_label"] = (df["fraud_probability"] >= 0.5).astype(int)

    shap_vals = compute_shap_matrix(model, X)
    df["shap_values"] = serialise_shap_rows(shap_vals)

    return {
        "rows": len(df),
        "data": df.head(50).to_dict(orient="records"),
    }
