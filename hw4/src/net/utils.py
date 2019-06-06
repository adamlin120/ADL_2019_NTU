import torch
from torch import nn


def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        try:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
        except AttributeError:
            pass


def get_label_id(hair, eye, face, glass):
    id = torch.argmax(glass.detach(), 1)
    id += torch.argmax(face.detach(), 1) * glass.size(1)
    id += torch.argmax(eye.detach(), 1) * glass.size(1) * face.size(1)
    id += torch.argmax(hair.detach(), 1) * glass.size(1) * face.size(1) * eye.size(1)
    return id
