pplication of Deep Learning 
# HW4 Conditional Generation of Cartoonset100k


### Train
```sh
$ python3.6 -m src.train path/to/model/folder/ 
```

### Test
```sh
$ python3.6 -m src.test path/to/model/folder/ CKPT_GLOBAL_STEP --label path/to/label.txt --output_dir dir/to/save/imgs/
```

### Plot GIF
```sh
$ python3.6 -m src.make_gif path/to/model/folder/ -k FIRST_K_IMAGES -d GID_FRAME_DURATION
```


### Packages
| Packages | Version |
| ------ | ------ |
| Python | 3.6 |
| CUDA | 10 |
| cuDNN | 7 |
| PyTorch | 1.1 |
| TorchVision | 0.3 |
| Python-Box |  |
| ipdb |  |
| tqdm |  |
| pyyaml |  |
| PIL |  |
| imageio |  |
