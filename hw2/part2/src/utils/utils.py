import os
import re
from pathlib import Path
from pprint import pprint
import logging
import yaml
from box import Box

import torch


def load_config(model_folder):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                           |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                           |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                           |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                           |[-+]?\\.(?:inf|Inf|INF)
                           |\\.(?:nan|NaN|NAN))$''', re.X),
                           list(u'-+0123456789.'))

    logging.info(f"Model Folder: {model_folder}")
    with open(os.path.join(model_folder, 'config.yml')) as f:
        config = Box(yaml.load(f, Loader=loader))
        config.loss.args.weight = torch.tensor(config.loss.args.weight)
    logging.info(f"Config: {pprint(config.to_dict())}")
    return config


def get_device(CUDA_DEVICE_STR):
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE_STR
    device = 'cuda' if CUDA_DEVICE_STR != '' else 'cpu'
    logging.info(f"Using device: {device}")
    return device


def save_model(path, epoch, model, optimizer):
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def drop_ckpt_optimizer(path):
    ckpt = torch.load(path)
    torch.save({
        'epoch': ckpt['epoch'],
        'model_state_dict': ckpt['model_state_dict']
        }, 
        path.split('.')[0]+'no_optim.pt')


def load_model(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    try:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except KeyError:
        pass
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def load_model_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint['model_state_dict']

