import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Cartoonset100kDataset(Dataset):
    def __init__(self, 
                 attr_file='../selected_cartoonset100k/cartoon_attr.txt', 
                 image_dir='../selected_cartoonset100k/images', 
                 transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])):
        self.attr_file = load_labels(attr_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.attr_file[idx][0])
        image = Image.open(img_name).convert('RGB')
        labels = torch.FloatTensor(list(map(float, self.attr_file[idx][1:])))

        if self.transform:
            image = self.transform(image)

        return image, labels


def load_labels(label):
    data = []
    with open(label, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                print('Number of images:',line)
            elif i == 1:
                print('Lables:')
                print('\n'.join(line.split(' ')))
            else:
                data.append(line.split(' '))
    return data
