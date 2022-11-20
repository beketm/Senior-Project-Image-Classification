from torchvision import models, transforms
import torch.nn as nn
import torch
import bentoml

PATH = "model_weights.pth"
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

if __name__=="__main__":

    model = get_model().cpu()
    print(len(model))
    model = nn.Module(model)

    bentoml.pytorch.save(
        model,
        "my_torch_model",
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )
