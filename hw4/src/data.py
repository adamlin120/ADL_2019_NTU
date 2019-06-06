import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Cartoonset100kDataset(Dataset):
    def __init__(self,
                 device,
                 pre_load=False,
                 attr_file='../selected_cartoonset100k/cartoon_attr.txt',
                 image_dir='../selected_cartoonset100k/images', 
                 transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])):
        self.device = device
        self.attr_file = load_labels(attr_file)
        self.image_dir = image_dir
        self.transform = transform
        self.pre_load = pre_load

        if self.pre_load:
           self.image = {i:self.transform(Image.open(os.path.join(self.image_dir, a[0])).convert('RGB'))
                          for i, a in enumerate(self.attr_file)}

    def __len__(self):
        return len(self.attr_file)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.image[idx]
        else:
            img_name = os.path.join(self.image_dir,
                                    self.attr_file[idx][0])
            image = self.transform(Image.open(img_name).convert('RGB'))
        labels = torch.FloatTensor(list(map(float, self.attr_file[idx][1:])))
        hair = torch.FloatTensor(labels[:6])
        eye = torch.FloatTensor(labels[6:10])
        face = torch.FloatTensor(labels[10:13])
        glass = torch.FloatTensor(labels[13:])

        return image, hair, eye, face, glass, labels


def load_labels(label):
    data = []
    print(f"Loading Label File: {label}")
    with open(label, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if i == 0:
                print('Number of images:',line)
            elif i == 1:
                print('Lables:')
                print('\n'.join(line.split(' ')))
            else:
                data.append(line.split(' '))
    # the return type if "list of str"
    # user should parse the numerical element
    return data


def get_all_label_combinations():
    labels = torch.zeros(6*4*3*2, 6+4+3+2)
    cnt = 0
    for h in range(6):
        for e in range(6, 6+4):
            for f in range(6+4, 6+4+3):
                for g in range(6+4+3, 6+4+3+2):
                    labels[cnt][h] = 1.0
                    labels[cnt][e] = 1.0
                    labels[cnt][f] = 1.0
                    labels[cnt][g] = 1.0
                    cnt += 1
    return (labels[:, :6],
            labels[:, 6:10],
            labels[:, 10:13],
            labels[:, 13:])
