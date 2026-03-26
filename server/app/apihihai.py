from fastapi import FastAPI, UploadFile, File
import tempfile
import pandas as pd
from app.api.service import run_feature_pipeline

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    # 🔥 LIMIT DATA (VERY IMPORTANT)
    df = pd.read_csv(path)
    df = df.head(1000)
    df.to_csv(path, index=False)

    result = run_feature_pipeline(csv_path=path)

    return result
