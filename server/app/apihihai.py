from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import pandas as pd
from app.api.service import run_feature_pipeline

app = FastAPI()

# ✅ FIX: allow frontend (Vercel) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    # ✅ LIMIT TO 20k ROWS
    df = pd.read_csv(path)
    df = df.head(20000)
    df.to_csv(path, index=False)

    result = run_feature_pipeline(csv_path=path)

    return result
