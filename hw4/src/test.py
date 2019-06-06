import sys
import os
import argparse
import random
from glob import glob
import ipdb
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image
from .train import parse_config


def parse_args():
    parser = argparse.ArgumentParser(description="ADL HW4 cGAN")
    parser.add_argument('model_folder', help='model folder with config file')
    parser.add_argument('step', help='select ckpt')
    parser.add_argument('--label', required=True, help='label file')
    parser.add_argument('--output_dir', default=None, help='output directory to save images')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('-d', '--device', default='cuda:0', help='pytorch device')
    return parser.parse_args()


def main(args, config):
    # Set random seed
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    # Select device to run model
    device = torch.device(args.device)  # GPU: "cuda:0" / CPU: "cpu"

    # load label
    with open(args.label, 'r') as f:
        f = f.read().strip().split('\n')
        num_images = int(f[0])
        labels = torch.Tensor([[float(e) for e in line.strip().split(' ')]
                for line in f[2:]]).to(device)

    # load ckpt
    ckpt = torch.load(glob(os.path.join(args.model_folder,
                           'ckpt',
                           f'{args.step}*.pt'))[-1])

    # init G and D
    generator = Generator(config.noise_dim + 4 * 16, config.base_acti_maps).to(device)
    generator.load_state_dict(ckpt['generator'])

    images = []
    for start in tqdm(np.arange(0, labels.size(0), args.batch_size)):
        gen_labels = (labels[start:start+args.batch_size, :6],
                      labels[start:start+args.batch_size, 6:6+4],
                      labels[start:start+args.batch_size, 6+4:6+4+3],
                      labels[start:start+args.batch_size, 6+4+3:])
        noise = torch.randn(gen_labels[0].size(0), config.noise_dim, device=device)
        with torch.no_grad():
            images.append(generator(noise, *gen_labels).cpu())
    assert sum([len(batch) for batch in images]) == num_images

    # save images
    output_dir = args.output_dir if args.output_dir else os.path.join(args.model_folder, 'test_images_' +
                                                                      f'{args.step}_' +
                                                                      os.path.basename(args.label).split('.')[0])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print(f"Saving Images to {output_dir} from label: {args.label}")
    i = 0
    for batch in tqdm(images):
        for image in batch:
            image_path = os.path.join(output_dir, f"{i}.png")
            save_image(image / 2 + 0.5, image_path)
            i += 1


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        config = parse_config(args.model_folder)
        if hasattr(config, 'arch'):
            if config.arch == 'noinpute':
                from .net.cGAN_noinpute import Generator
            elif config.arch == 'condbn':
                from .net.cGAN_condbn import Generator
            elif config.arch == 'dis_drop':
                from .net.cGAN_dis_dropout import Generator
            else:
                raise Exception(f"Unknown model arch: {config.arch}")
            print(f"Using architecture: {config.arch}")
        else:
            from .net.cGAN_projection import Generator
            print('Using default architecture')
        main(args, config)
