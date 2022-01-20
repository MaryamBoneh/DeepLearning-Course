import torch, torchvision
import torch.nn as nn


def Model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

    return model