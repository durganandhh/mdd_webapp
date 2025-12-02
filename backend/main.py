from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from inference import run_inference
from fastapi.staticfiles import StaticFiles
import os
import uuid

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/score")
async def score_audio(
    transcript: str = Form(...),
    audio: UploadFile = File(...)
):
 
    file_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{file_id}.wav"

    with open(file_path, "wb") as f:
        f.write(await audio.read())


    result = run_inference(file_path, transcript)

    return {
        "status": "success",
        "prediction": result
    }
