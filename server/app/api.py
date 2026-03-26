from fastapi import FastAPI, UploadFile, File
import tempfile
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

    result = run_feature_pipeline(csv_path=path)

    return result
