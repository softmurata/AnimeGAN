# anime GAN
# argument module
# Create animeGAN Network
# Dataset loader
# Dataset module

# python preprocess.py --img_size 256
# python main.py --gpu_number 0 

# import module
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import AnimeGAN


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--resume_g_path', type=str, default='', help='ex) ./results/models/exp1/checkpoint_generator_035000.pth.tar')
parser.add_argument('--dataset_dir', type=str, default='./dataset/')
parser.add_argument('--anime_name', type=str, default='sentochihiro')
parser.add_argument('--exp_name', type=str, default='exp1')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--pretrained_epoch', type=int, default=10)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--loss_func_type', type=str, default="lsgan", help="wgan-lp, wgan-gp, lsgan, dragan, hinge")

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


# build anime GAN model
phase = 'train'
animegan = AnimeGAN(args, phase, device)

if phase == 'train':
    animegan.train()

else:
    input_image_path = './dataset/test/005000.png'
    real_image = Image.open(input_image_path).convert("RGB")
    real_image = transforms.ToTensor(real_image)
    real_image = real_image.view(1, real_image.shape[0], real_image.shape[1], real_image.shape[2])  # (1, c, h, w)
    real_image = real_image.to(device)  # for cuda
    generated_images = animegan.predict(real_image)

    gene_image = (generated_images[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2

    save_dir = './out_images/'
    os.makedirs(save_dir)
    save_path = save_dir + input_image_path.split('/')[-1]
    plt.imsave(save_path, gene_image) 

