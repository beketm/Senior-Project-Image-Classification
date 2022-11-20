
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle 
import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Image Classifier using Machine Learning')
# st.text('Upload the Image from the listed category.\n[Rose, Cricket Bat, Icecream Cone, Covid Vaccine, Chocolate]')

PATH = "model_weights.pth"
device = "cpu"
class_names=['akorda', 'baiterek', 'khanshatyr', 'mangilikel', 'mosque', 'nuralem', 'piramida', 'shabyt']
num_classes = len(class_names)

def get_model():
    model_ft = models.mobilenet_v2(pretrained=True)    

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.6, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
    )
    model_ft.load_state_dict(torch.load(PATH))
    model_ft.eval()
    return model_ft


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
    tensor = tensor.to(device)
    output = model(tensor)
     
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), class_names[classes.item()]

model = get_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png", "jpeg", "JPEG"])
if uploaded_file is not None:
  img = Image.open(uploaded_file).convert("RGB")
  st.image(img,caption='Uploaded Image')

  if st.button('PREDICT'):

    # st.write('Result...')
    # flat_data=[]
    # img = np.array(img)
    # img_resized = resize(img,(150,150,3))
    # flat_data.append(img_resized.flatten())
    # flat_data = np.array(flat_data)
    # y_out = model.predict(flat_data)
    # y_out = Categories[y_out[0]]
    prediction_prob, prediction = get_prediction(img)
    st.title(f' PREDICTED OUTPUT: {prediction.upper()}')
    st.title(f' PREDICTION PROBABILITY: {prediction_prob:.3f}')
    # q = model.predict_proba(flat_data)
    # for index, item in enumerate(Categories):
    #   st.write(f'{item} : {q[0][index]*100}%')

st.text("")
st.text('Made by Beket Myrzanov')
