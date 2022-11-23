import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from torchvision import models, transforms
import torch.nn as nn
import torch
from PIL import Image
from io import BytesIO

PATH = "model_weights.pth"
device = "cpu"
class_names = ['akorda', 'baiterek', 'khanshatyr',
               'mangilikel', 'mosque', 'nuralem', 'piramida', 'shabyt']
num_classes = len(class_names)


def get_model():
    model_ft = models.mobilenet_v2(pretrained=True)

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.6, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    )
    model_ft.load_state_dict(torch.load(PATH))
    model_ft.eval()
    return model_ft


app = FastAPI()
model = get_model()


def transform_image(image):
    my_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    return my_transforms(image).unsqueeze(0)


def get_prediction(image):
    tensor = transform_image(image)
    print(tensor.shape)
    tensor = tensor.to(device)
    output = model(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), class_names[classes.item()]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    return image


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())

    prediction = get_prediction(image)
    response = {"probability": prediction[0], "class": prediction[1]}
    return response


@app.get("/")
async def health_check():
    return {"message": "ok"}
