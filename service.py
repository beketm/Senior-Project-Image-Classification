import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import inference
import base64
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

infer = inference.Inference()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    # return {"filename": file.filename}
    try:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        prediction = infer.classify_image(await file.read())
        print(prediction)
        return prediction
    except Exception as e:
        return {"error": e.__str__()}


class Photo(BaseModel):
    base64: str


@app.post("/predict/image_json")
async def predict_api_json(body: Photo):
    # return {"filename": file.filename}
    try:
        content = base64.b64decode(body.base64)
        prediction = infer.classify_image(content)
        print(prediction)
        return prediction
    except Exception as e:
        return {"error": e.__str__()}


@app.get("/")
async def health_check():
    return {"message": "ok"}
