import sys
import os
import shutil
import re
import argparse
import random
import math
import yaml
import ipdb
from box import Box
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .data import Cartoonset100kDataset, get_all_label_combinations
from .net.utils import weight_init


def parse_args():
    parser = argparse.ArgumentParser(description="ADL HW4 cGAN")
    parser.add_argument('model_folder', help='model folder with config file')
    parser.add_argument('-d', '--device', default='cuda:0', help='pytorch device')
    parser.add_argument('--online_loading', action='store_true')
    parser.add_argument('--display_freq', default=50, type=int)
    parser.add_argument('--save_img_freq', default=500, type=int)
    parser.add_argument('--save_model_freq', default=10000, type=int)
    return parser.parse_args()


def parse_config(model_folder):
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

    with open(os.path.join(model_folder, 'config.yaml')) as f:
        return Box(yaml.load(f, Loader=loader))


def get_random_one_hot_label(batch_size, label_size):
    ret = []
    for _ in range(batch_size):
        k = random.randint(0, label_size - 1)
        ret.append([0] * k + [1] + [0] * (label_size - 1 - k))
    ret = torch.FloatTensor(ret)
    return ret


def main(args, config):
    # Set random seed
    global gen_loss
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    # Select device to run model
    device = torch.device(args.device)  # GPU: "cuda:0" / CPU: "cpu"
    print(f"Using device: {device}")

    # Dataset and Data Loader
    dataset = Cartoonset100kDataset(device,
                                    pre_load=(not args.online_loading),
                                    attr_file=config.attr_file,
                                    image_dir=config.image_dir)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False, num_workers=6)

    # init G and D
    generator = Generator(config.noise_dim + 4 * 16, config.base_acti_maps).to(device)
    generator.apply(weight_init)
    discriminator = Discriminator(config.base_acti_maps).to(device)
    discriminator.apply(weight_init)

    labels_fixed = tuple(map(lambda x: x.to(device),
                             get_all_label_combinations()))
    num_sample = labels_fixed[0].size(0)
    noise_fixed = torch.randn(num_sample, config.noise_dim, device=device)
    # noise_fixed = torch.empty(NUM_SAMPLE, config.noise_dim, device=device, dtype=torch.float).uniform_(-1, 1)

    valid_label = 1
    fake_label = 0

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    auxiliary_loss = nn.CrossEntropyLoss().to(device)

    optimizer = Box({'Gen': optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)),
                     'Dis': optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))})

    global_step = 0
    for epoch in range(config.num_epoch):
        for i, (real_imgs, hairs, eyes, faces, glasses, labels) in enumerate(loader):
            global_step += 1
            real_imgs, hairs, eyes, faces, glasses = \
                real_imgs.to(device), hairs.to(device), eyes.to(device), faces.to(device), glasses.to(device)
            batch_size = real_imgs.size(0)
            # Adversarial ground truths
            valid = torch.FloatTensor(batch_size, 1).fill_(valid_label).to(device)
            # add noise to labels
            valid = valid + torch.empty_like(valid, device=device, dtype=torch.float).uniform_(-0.3, 0.3)
            fake = torch.FloatTensor(batch_size, 1).fill_(fake_label).to(device)
            # add noise to labels
            fake = fake + torch.empty_like(fake, device=device, dtype=torch.float).uniform_(0, 0.3)

            # Train discriminator
            optimizer.Dis.zero_grad()
            # loss for all read imgs
            real_labels = (hairs, eyes, faces, glasses)
            # real_imgs = real_imgs + torch.randn_like(real_imgs) * 0.4 / epoch
            validity_real, real_hair_val, real_eye_val, real_face_val, real_glass_val = discriminator(real_imgs,
                                                                                                      *real_labels)
            if hasattr(config, 'adversarial_loss'):
                if config.adversarial_loss == 'BCE':
                    dis_real_adv_loss = adversarial_loss(validity_real, valid)
                else:
                    raise Exception(f"Unknown adversarial loss: {config.adversarial_loss}")
            else:
                dis_real_adv_loss = (-1) * torch.mean(validity_real)
            dis_real_loss = config.loss_weight.adv * dis_real_adv_loss + \
                config.loss_weight.hair * auxiliary_loss(real_hair_val, torch.argmax(real_labels[0], 1)) + \
                config.loss_weight.eye * auxiliary_loss(real_eye_val, torch.argmax(real_labels[1], 1)) + \
                config.loss_weight.face * auxiliary_loss(real_face_val, torch.argmax(real_labels[2], 1)) + \
                config.loss_weight.glass * auxiliary_loss(real_glass_val, torch.argmax(real_labels[3], 1))
            # loss for all fake imgs
            # sample noise
            noise = torch.randn(batch_size, config.noise_dim, device=device)
            # noise = torch.empty(batch_size, config.noise_dim, device=device, dtype=torch.float).uniform_(-1, 1)
            # generate imgs from noise
            gen_labels = (get_random_one_hot_label(*hairs.size()).to(device),
                          get_random_one_hot_label(*eyes.size()).to(device),
                          get_random_one_hot_label(*faces.size()).to(device),
                          get_random_one_hot_label(*glasses.size()).to(device))
            gen_imgs = generator(noise, *gen_labels)
            validity_fake, fake_hair_val, fake_eye_val, fake_face_val, fake_glass_val = discriminator(gen_imgs.detach(),
                                                                                                      *gen_labels)
            if hasattr(config, 'adversarial_loss'):
                if config.adversarial_loss == 'BCE':
                    dis_fake_adv_loss = adversarial_loss(validity_fake, fake)
                else:
                    raise Exception(f"Unknown adversarial loss: {config.adversarial_loss}")
            else:
                dis_fake_adv_loss = torch.mean(validity_fake)
            dis_fake_loss = config.loss_weight.adv * dis_fake_adv_loss + \
                config.loss_weight.hair * auxiliary_loss(fake_hair_val, torch.argmax(gen_labels[0], 1)) + \
                config.loss_weight.eye * auxiliary_loss(fake_eye_val, torch.argmax(gen_labels[1], 1)) + \
                config.loss_weight.face * auxiliary_loss(fake_face_val, torch.argmax(gen_labels[2], 1)) + \
                config.loss_weight.glass * auxiliary_loss(fake_glass_val, torch.argmax(gen_labels[3], 1))
            dis_fake_loss = config.loss_weight.adv * dis_fake_adv_loss
            dis_loss = (dis_real_loss + dis_fake_loss) / 2
            dis_loss.backward()
            optimizer.Dis.step()

            if i % config.n_critic == 0:
                # Train Generator
                optimizer.Gen.zero_grad()
                # sample noise
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                # noise = torch.empty(batch_size, config.noise_dim, device=device, dtype=torch.float).uniform_(-1, 1)
                # generate imgs from noise
                gen_labels = (get_random_one_hot_label(*hairs.size()).to(device),
                              get_random_one_hot_label(*eyes.size()).to(device),
                              get_random_one_hot_label(*faces.size()).to(device),
                              get_random_one_hot_label(*glasses.size()).to(device))
                gen_imgs = generator(noise, *gen_labels)
                gen_validity, gen_hair_val, gen_eye_val, gen_face_val, gen_glass_val = \
                    discriminator(gen_imgs, *gen_labels)
                if hasattr(config, 'adversarial_loss'):
                    if config.adversarial_loss == 'BCE':
                        gen_adv_loss = adversarial_loss(gen_validity, valid)
                    else:
                        raise Exception(f"Unknown adversarial loss: {config.adversarial_loss}")
                else:
                    gen_adv_loss = (-1) * torch.mean(gen_validity)
                gen_loss = config.loss_weight.adv * gen_adv_loss + \
                    config.loss_weight.hair * auxiliary_loss(gen_hair_val, torch.argmax(gen_labels[0], 1)) + \
                    config.loss_weight.eye * auxiliary_loss(gen_eye_val, torch.argmax(gen_labels[1], 1)) + \
                    config.loss_weight.face * auxiliary_loss(gen_face_val, torch.argmax(gen_labels[2], 1)) + \
                    config.loss_weight.glass * auxiliary_loss(gen_glass_val, torch.argmax(gen_labels[3], 1))
                gen_loss.backward()
                optimizer.Gen.step()

            # display info to std out
            if i % args.display_freq == 0:
                print(f"GlobalStep: {global_step} Epoch:{epoch} Batch:{i} "
                      f"Gen Loss:{gen_loss.item()} "
                      f"Dis Loss:{dis_loss.item()} "
                      f"Dis Real Loss:{dis_real_loss.item()} Dis Fake Loss:{dis_fake_loss.item()}")

            # save images
            if global_step % args.save_img_freq == 0:
                with torch.no_grad():
                    gen_imgs = generator(noise_fixed, *labels_fixed)
                image_folder = os.path.join(args.model_folder, 'image/')
                if not os.path.isdir(image_folder):
                    os.mkdir(image_folder)
                image_path = os.path.join(image_folder, f"generated_{global_step}_{epoch}_{i}.png")
                save_image((gen_imgs.data - torch.min(gen_imgs.data)) /
                           torch.max(gen_imgs.data - torch.min(gen_imgs.data)),
                           image_path, nrow=int(math.ceil(math.sqrt(noise_fixed.size(0)))))
                shutil.copy(image_path, os.path.join(image_folder, 'live.png'))

            # save model checkpoint
            if global_step % args.save_model_freq == 0:
                ckpt_folder = os.path.join(args.model_folder, 'ckpt/')
                if not os.path.isdir(ckpt_folder):
                    os.mkdir(ckpt_folder)
                ckpt_path = os.path.join(ckpt_folder, f"{global_step}_{epoch}_{i}.pt")
                torch.save({'global_step': global_step,
                            'epoch': epoch,
                            'local_step': i,
                            'generator': generator.state_dict(),
                            'discriminator': discriminator.state_dict()}, ckpt_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        args = parse_args()
        config = parse_config(args.model_folder)

        if hasattr(config, 'loss_weight'):
            config.loss_weight = Box({'adv': config.loss_weight[0], 
                                  'hair': config.loss_weight[1], 
                                  'eye': config.loss_weight[2],
                                  'face': config.loss_weight[3],
                                  'glass': config.loss_weight[4]})
        else:
            config.loss_weight = Box({'adv': 0.01, 'hair': 0.2, 'eye': 0.2,
                                  'face': 0.2, 'glass': 0.2})
        if hasattr(config, 'arch'):
            if config.arch == 'noinpute':
                from .net.cGAN_noinpute import Generator, Discriminator
            elif config.arch == 'condbn':
                from .net.cGAN_condbn import Generator, Discriminator
            elif config.arch == 'dis_drop':
                from .net.cGAN_dis_dropout import Generator, Discriminator
            print(f"Using architecture: {config.arch}")
        else:
            from .net.cGAN_projection import Generator, Discriminator
            print('Using default architecture')
        main(args, config)

