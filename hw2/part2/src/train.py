import os
import traceback
import sys
import logging
import argparse

from tqdm import tqdm
import ipdb
from box import Box
import torch
from torch import nn
from warmup_scheduler import GradualWarmupScheduler

from .model import net
from .dataset import get_loader
from .metrics.accuracy import Accuracy
from .utils.utils import load_config, get_device, save_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str)
    args_parsed = parser.parse_args()
    return args_parsed


def set_logging(model_folder):
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(f"{os.path.join(model_folder, 'program_logging.log')}"),
                            logging.StreamHandler()
                        ])


def run_epoch(model, optimizer, criterion, loader, train, metrics, max_norm=-1):
    val = not train
    avg_loss = 0.0
    for metric in metrics:
        metric.reset()
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Train' if train else 'Val')
    if val:
        model.eval()
        model.to_eval()
    else:
        model.train()
        model.to_train()
    with torch.set_grad_enabled(train):
        for local_step, (x, y, mask, type_ids) in pbar:
            x, y = x.to(device), y.to(device)
            mask = mask.to(device) if mask is not None else None
            type_ids = type_ids.to(device)

            logits = model(x, attention_mask=mask, token_type_ids=type_ids)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if max_norm!=-1:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

            avg_loss = (avg_loss * local_step + loss.detach().cpu().item()) / (local_step+1)
            for metric in metrics:
                metric.update(logits, y)

            pbar.set_postfix(loss=f"{avg_loss:.8f}",
                             **{metric.__class__.__name__: str(metric) for metric in metrics})

    return avg_loss, metrics


def log_metrics(epoch, model_folder, loss, metrics):
    path = os.path.join(model_folder, 'log.csv')
    with open(path, 'a') as f:
        f.write(f"Epoch:{epoch}\n")
        f.write(f"loss,train,{loss.train},val,{loss.val}\n")
        for metric_train, metric_val in zip(metrics.train, metrics.val):
            f.write(f"{metric_train.__class__.__name__},train,{metric_train()},val,{metric_val()}\n")


def train(args, config, loader, device):
    logging.info('Start training...')
    model = getattr(net, config.model.name)(**config.model.args, **config.embedder)
    model = model.to(device)

    criterion = getattr(nn, config.loss.name)(**config.loss.args).to(device)
    optimizer = getattr(torch.optim, config.optimizer.name)(model.parameters(), **config.optimizer.args)
    if hasattr(config, 'lr_scheduler'):
        if hasattr(config.lr_scheduler, 'name'):
            scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
        else:
            scheduler = None
        if hasattr(config.lr_scheduler, 'warm_up'):
            scheduler_warm_up = GradualWarmupScheduler(optimizer, 
                                                       multiplier=config.lr_scheduler.warm_up.multiplier, 
                                                       total_epoch=config.lr_scheduler.warm_up.epoch, 
                                                       after_scheduler=scheduler)


    loss = Box({'train': 0.0, 'val': 0.0})
    metrics = Box({'train': [Accuracy()], 'val': [Accuracy()]})

    for epoch in range(config.train.n_epoch):
        if hasattr(config, 'lr_scheduler'):
            if hasattr(config.lr_scheduler, 'warm_up'):
                scheduler_warm_up.step()
            else:
                scheduler.step()

        loss.train, metrics.train = run_epoch(model, optimizer, criterion, loader.train,
                                              train=True, metrics=metrics.train,
                                              max_norm=config.max_norm if hasattr(config, 'max_norm') else -1)
        loss.val, metrics.val = run_epoch(model, optimizer, criterion, loader.val,
                                          train=False, metrics=metrics.val)

        saved_path = os.path.join(args.model_folder, 'checkpoints', f'epoch_{epoch}.pt')
        save_model(saved_path, epoch, model, optimizer)
        log_metrics(epoch, args.model_folder, loss, metrics)


if __name__ == '__main__':
    try:
        args = parse_args()
        set_logging(args.model_folder)
        config = load_config(args.model_folder)
        device = get_device(config.CUDA_DEVICE)
        loader = get_loader(config)
        train(args, config, loader, device)

    except KeyboardInterrupt:
        pass

    except BaseException:
        exc_type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
