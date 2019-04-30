import traceback
import sys
import os
import argparse

from tqdm import tqdm
import ipdb
import pandas as pd
import torch

from .model import net
from .dataset import get_pred_loader
from .utils.utils import load_config, get_device, load_model, load_model_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', nargs='+', type=str)
    parser.add_argument('-e', '--epoch', nargs='+', type=str)
    parser.add_argument('--csv', type=str, default='./data/classification/test.csv')
    parser.add_argument('-g', '--CUDA_DEVICE', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default=None)
    args_parsed = parser.parse_args()
    return args_parsed


def change_config_batch_size(config, batch_size):
    config.train.batch_size = batch_size
    config.val.batch_size = batch_size
    return config


def predict_dataset(model, loader, prob=False):
    pred = []
    model.eval()
    model.to_eval()
    model.fine_tune_embedder = False
    with torch.no_grad():
        for local_step, (x, _, mask, type_id) in tqdm(enumerate(loader), total=len(loader), desc='Predict'):
            x = x.to(device)
            mask = mask.to(device) if mask is not None else None
            type_id = type_id.to(device)

            logits = model(x, mask, type_id)
            if prob:
                pred.append(torch.softmax(logits.cpu(), 1))
            else:
                pred.append(torch.argmax(logits.cpu(), 1))
    pred = torch.cat(pred)
    return pred


def write_prediction(pred, args):
    df = pd.read_csv(args.csv).drop(columns='text')
    df.label = pred + 1
    prediction_folder = os.path.join(args.model_folder, 'prediction')
    if not os.path.isdir(prediction_folder):
        os.mkdir(prediction_folder)
    csv_path = os.path.join(prediction_folder, f'epoch_{args.epoch}.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file written at {csv_path}")
    if args.output is not None:
        df.to_csv(args.output, index=False)
        print(f"CSV file written at {args.output}")



def write_ensemble_prediction(pred, args):
    pred = torch.argmax(pred, 1) + 1
    df = pd.read_csv(args.csv).drop(columns='text')
    df.label = pred
    prediction_folder = os.path.join('saved', 'ensemble_prediction')
    if not os.path.isdir(prediction_folder):
        os.mkdir(prediction_folder)
    csv_path = os.path.join(
            prediction_folder, 
            '-'.join([f'{os.path.basename(os.path.normpath(model_folder))}_epoch_{epoch}' 
                      for model_folder, epoch in zip(args.model_folder, args.epoch)]) + '.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file written at {csv_path}")
    if args.output is not None:
        df.to_csv(args.output, index=False)
        print(f"CSV file written at {args.output}")


if __name__ == '__main__':
    try:
        args = parse_args()
        device = get_device(args.CUDA_DEVICE)

        if len(args.model_folder)==1:
            args.model_folder = args.model_folder[0]
            args.epoch = args.epoch[0]

            config = load_config(args.model_folder)
            config = change_config_batch_size(config, args.batch_size)
            loader = get_pred_loader(args.csv, config)
            model_state = load_model_state_dict(os.path.join(args.model_folder, f"checkpoints/epoch_{args.epoch}.pt"))
            model = getattr(net, config.model.name)(**config.model.args, **config.embedder)
            model = model.to(device)
            model, _, _ = load_model(os.path.join(args.model_folder, f'checkpoints/epoch_{args.epoch}.pt'), model)
            pred = predict_dataset(model, loader)
            write_prediction(pred, args)
        else:
            preds = []
            for model_folder, epoch in zip(args.model_folder, args.epoch):
                config = load_config(model_folder)
                config = change_config_batch_size(config, args.batch_size)
                loader = get_pred_loader(args.csv, config)
                model = getattr(net, config.model.name)(**config.model.args, **config.embedder)
                model = model.to(device)
                model, _, _ = load_model(os.path.join(model_folder, f'checkpoints/epoch_{epoch}.pt'), model)
                pred = predict_dataset(model, loader, prob=True)
                preds.append(pred)
            pred = torch.stack(preds).mean(0)   
            write_ensemble_prediction(pred, args)


    except KeyboardInterrupt:
        pass

    except BaseException:
        exc_type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
