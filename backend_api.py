from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import subprocess

app = FastAPI()

# Allow frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
GEMINI_KEY = "AIzaSyBIKd_rGYsfC2sAsOJDAstEaF1UDzEN58k"  # Replace with env var for prod
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_ordered.pdf")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Run the PDFReconstructor script
    try:
        result = subprocess.run([
            "python", "pdf_reconstructor.py",
            input_path, output_path,
            "--llm-provider", "gemini",
            "--llm-key", GEMINI_KEY
        ], capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    # Return the processed PDF
    return FileResponse(output_path, filename="ordered.pdf", media_type="application/pdf") 