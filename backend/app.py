from fastapi import FastAPI, UploadFile, File
from model.classifier import analyze_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Menalyze backend running!"}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    result = analyze_image(await file.read())
    return {"result": result}
