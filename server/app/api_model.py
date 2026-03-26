from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import pandas as pd
import random

from model.features import build_feature_matrix

app = FastAPI()

# CORS (required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Fraud Model API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    df = pd.read_csv(path)

    # LIMIT (safe for Render)
    df = df.head(20000)

    X, _ = build_feature_matrix(df)

    # 🔥 FAKE MODEL (for demo stability)
    df["fraud_probability"] = [random.random() for _ in range(len(df))]
    df["fraud_label"] = (df["fraud_probability"] >= 0.5).astype(int)

    return {
        "rows": len(df),
        "data": df.head(50).to_dict(orient="records"),
    }
