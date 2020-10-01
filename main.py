# anime GAN
# argument module
# Create animeGAN Network
# Dataset loader
# Dataset module

# python preprocess.py --img_size 256
# python main.py --gpu_number 0 

# import module
import argparse
import os
import torch
from model import AnimeGAN
from dataset import data_load

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--resume_g_path', type=str, default='', help='ex) ./results/models/exp1/checkpoint_generator_035000.pth.tar')
parser.add_argument('--dataset_dir', type=str, default='/raid/murata/AnimeGAN/datasets/')
parser.add_argument('--out_image_dir', type=str, default='/raid/murata/AnimeGAN/out_images/')
parser.add_argument('--anime_name', type=str, default='sentochihiro')
parser.add_argument('--exp_name', type=str, default='exp1')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--pretrained_epoch', type=int, default=2)
parser.add_argument('--training_rate', type=int, default=1)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--loss_func_type', type=str, default="lsgan", help="wgan-lp, wgan-gp, lsgan, dragan, hinge")
parser.add_argument('--vgg_pretrained_weights', type=str, default='./vgg_weights/vgg19-dcbb9e9d.pth')

parser.add_argument('--lr_g', type=float, default=8e-5)
parser.add_argument('--lr_d', type=float, default=1.6e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)

parser.add_argument('--con_weight', type=float, default=1.5)
parser.add_argument('--style_weight', type=float, default=3)
parser.add_argument('--color_weight', type=float, default=10)
parser.add_argument('--g_adv_weight', type=float, default=300)
parser.add_argument('--d_adv_weight', type=float, default=300)
args = parser.parse_args()

device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else "cpu"

os.makedirs(args.out_image_dir, exist_ok=True)

torch.backends.cudnn.enabled = False

# build anime GAN model
phase = 'train'
animegan = AnimeGAN(args, phase, device)

if phase == 'train':
    animegan.train()

else:
    animegan.predict()

