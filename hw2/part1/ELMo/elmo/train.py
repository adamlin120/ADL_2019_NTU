import sys
import logging
import traceback
import ipdb
import argparse
import os
import yaml
import multiprocessing
from box import Box
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .dataset import ElmoDataset
from . import net


PAD_INDEX = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-g', '--gpu', default='', type=str)
    parser.add_argument('-c', '--cpu', default=-1, type=int)
    args_parsed = parser.parse_args()
    if args_parsed.cpu == -1:
        args_parsed.cpu = multiprocessing.cpu_count() // 2
    return args_parsed


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')
    with open(os.path.join(args.model, 'config.yml'), 'r') as stream:
        config = yaml.load(stream)
        config = Box(config)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"using device: {device}")

        logging.info('loading dataset...')
        dataset = Box({'train': ElmoDataset(config.data.train, config.data.idx2word),
                       'val': ElmoDataset(config.data.val, config.data.idx2word)})

        logging.info('creating dataloader...')
        loader = Box({'train': DataLoader(dataset['train'], batch_size=config.train.batch_size, shuffle=True,
                                          num_workers=args.cpu, collate_fn=ElmoDataset.collect_fn, pin_memory=True),
                      'val': DataLoader(dataset['val'], batch_size=config.train.batch_size,
                                        num_workers=args.cpu, collate_fn=ElmoDataset.collect_fn, pin_memory=True)})

        logging.info(f'constructing net: {config.model.net}')
        net = getattr(net, config.model.net)(config)

        optimizer = getattr(torch.optim, config.train.optimizer.name)(
            net.parameters(), **config.train.optimizer.params)
        if config.train.scheduler.enable:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config.train.scheduler.milestones,
                                                             gamma=config.train.scheduler.gamma)

        logging.info('start training...')
        # model move to device
        net = net.to(device)
        global_step = 0
        log_path = os.path.join(args.model, 'log.txt')
        for epo in range(config.train.epoch):
            train_iter = tqdm(loader.train, desc="Train")
            net.train()
            avg_loss = 0
            run_loss, run_f_loss, run_b_loss, run_accu = [], [], [], []
            for local_step, batch in enumerate(train_iter):
                global_step += 1
                optimizer.zero_grad()
                # data move to device
                batch = ElmoDataset.move_batch_to_device(batch, device)

                f_loss, b_loss, accu = net(batch)
                loss = (f_loss + b_loss) / 2
                loss.backward()

                if config.train.clip_grad.enable:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.train.clip_grad.norm)
                optimizer.step()

                avg_loss = (avg_loss * local_step + loss.detach().cpu()) / (local_step+1)
                run_loss.append(loss.detach().cpu())
                run_accu.append(accu)

                if len(run_loss)>300:
                    del run_loss[0]
                    del run_accu[0]
                if global_step % config.train.display_step == 0:
                    perplexity = torch.exp(sum(run_loss)/len(run_loss))
                    train_iter.set_postfix_str(f"loss: {(sum(run_loss)/len(run_loss)):.6f} "
                                               f"perplexity: {perplexity:.3f} "
                                               f"accu: {(sum(run_accu)/len(run_accu)):.3f}")

                if global_step % config.val.val_step == 0:
                    with open(log_path, 'a') as f:
                        f.write(f"epoch:{epo},step:{global_step}\n")
                        f.write(f"train,loss,{avg_loss},perplexity,{torch.exp(avg_loss)}\n")
                    if config.train.scheduler.enable:
                        scheduler.step()
                    val_iter = tqdm(loader.val, desc='Val')
                    net.eval()
                    avg_loss = 0
                    run_loss, run_f_loss, run_b_loss, run_accu = [], [], [], []
                    with torch.no_grad():
                        for step, batch in enumerate(val_iter):
                            batch = ElmoDataset.move_batch_to_device(batch, device)
                            f_loss, b_loss, accu = net(batch)
                            loss = (f_loss + b_loss) / 2
                            run_loss.append(loss.detach().cpu())
                            run_accu.append(accu)
                            perplexity = torch.exp(sum(run_loss) / len(run_loss))
                            avg_loss = (avg_loss * step + loss.detach().cpu()) / (step+1)
                            val_iter.set_postfix_str(
                                f"loss: {sum(run_loss) / len(run_loss):.6f} "
                                f"perplexity: {perplexity:.3f} "
                                f"accu: {sum(run_accu) / len(run_accu):.3f}"
                            )
                    logging.info(f"Validation: Loss {sum(run_loss) / len(run_loss)}")
                    with open(log_path, 'a') as f:
                        f.write(f"val,loss,{avg_loss},perplexity,{torch.exp(avg_loss)}\n")
                    # save model
                    net.eval()
                    torch.save({
                        'epoch': epo,
                        'step': global_step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                        os.path.join(args.model,
                                     f'checkpoint_{epo}_{global_step}_{perplexity:.6f}.pt')
                    )
                    net.train()
                    run_loss, run_f_loss, run_b_loss, run_accu = [], [], [], []
        net.eval()
    except KeyboardInterrupt:
        pass
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
