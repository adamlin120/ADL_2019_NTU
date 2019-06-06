import os
import sys
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import ipdb
import numpy
import imageio
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


parser = ArgumentParser()
parser.add_argument("model_folder")
parser.add_argument("-k", default=40, type=int)
parser.add_argument("-d", '--duration', default=1, type=int)
args = parser.parse_args()

image_folder = os.path.join(args.model_folder, 'image')
gif_path = os.path.join(args.model_folder, 'train_progress.gif')
img_path_list = glob(os.path.join(image_folder, '*_*_*.png'))
img_path_list = list(sorted(img_path_list, key=lambda path: int(path.split('_')[-3])))[::7][:args.k]


imgs = []
for img_path in tqdm(img_path_list):
    img = imageio.imread(img_path)
    iteration = int(img_path.split('_')[-3])

    blank_image = Image.new('RGB', (img.shape[1], 100))
    draw = ImageDraw.Draw(blank_image)
    font = ImageFont.truetype("./src/arial.ttf", 25)
    draw.text((img.shape[1]//2 - 30, 40), f"Iteration: {iteration}",(167,255,255),font=font)

    blank_image = numpy.array(blank_image)
    img = numpy.concatenate((blank_image, img), 0)
    imgs.append(img)

assert len(imgs) == args.k

imageio.mimsave(gif_path, imgs, format='GIF', duration=args.duration)

