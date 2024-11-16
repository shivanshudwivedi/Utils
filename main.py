from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ml_model import ObjectDetectionModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = ObjectDetectionModel()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    results = model.process_image(contents)
    return {"detections": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)