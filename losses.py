import numpy as np
import torch
import torch.nn as nn
# ToDo: implement con_sty_loss(), color_loss(), generator_loss(), discriminator_loss()
# CartoonGAN => (real, fake), ()
# import torchvision.models as models
from model import VGG19
from utils import rgb2yuv


def gram(feature_map):
    # feature map = (batch_size, c, h, w)
    feature_map = feature_map.reshape(feature_map.shape[0], feature_map.shape[1], -1)  # (batch_size, channel, height * width)
    feature_map_trans = feature_map.permute((0, 2, 1))  # numpy.transpose
    norm = (np.product(feature_map.shape)) / feature_map.shape[0]
    feature_map_mat = torch.matmul(feature_map_trans, feature_map)  # (batch_size, channel, channel)
    return feature_map_mat / norm

def con_sty_loss(vgg, real_image, anime_gray_d, real_g):

    l1_loss = nn.L1Loss()

    vgg = VGG19(pretrained_weights, feature_mode)
    # get feature map
    real_image_feature_map = vgg(real_image)
    anime_gray_feature_map = vgg(anime_gray_d)
    fake_feature_map = vgg(real_g)

    con_loss = l1_loss(real_image_feature_map, fake_feature_map)
    style_loss = l1_loss(gram(anime_gray_feature_map), gram(fake_feature_map))

    return con_loss, style_loss
    

def color_loss(real_image, real_g):
    # loss function
    l1_loss = nn.L1Loss()
    huber_loss = nn.SmoothL1Loss()
    # real image and fake image
    # rgb to yuv
    real_yuv = rgb2yuv(real_image)
    fake_yuv = rgb2yuv(real_g)

    # y channel loss(L1 Loss)
    y_ch_loss = l1_loss(fake_yuv[:, 0, :, :], real_yuv[:, 0, :, :])
    # u, v, channel loss(Huber Loss)
    u_ch_loss = huber_loss(fake_yuv[:, 1, :, :], real_yuv[:, 1, :, :])
    v_ch_loss = huber_loss(fake_yuv[:, 2, :, :], real_yuv[:, 2, :, :])

    return y_ch_loss + u_ch_loss + v_ch_loss 


def generator_loss(loss_func_type, real_d):
    # real_d = D(G(x))
    # BCE loss
    fake_loss = 0

    bce_loss = nn.BCELoss()

    if loss_func_type in ['wgan-lp', 'wgan-gp']:
        fake_loss = -torch.mean(real_d)
    if loss_func_type == 'lsgan':
        fake_loss = torch.mean(torch.square(real_d - 1.0))
    if loss_func_type in ['gan', 'dragan']:
        fake_loss = torch.mean(bce_loss(real_d, torch.ones_like(real_d)))
    if loss_func_type == 'hinge':
        fake_loss = -torch.mean(real_d)

    return fake_loss


def discriminator_loss(loss_func_type, anime_d, anime_gray_d, real_d, anime_smooth_d):
    # real, gray, fake, real_blur
    # BCE loss
    # loss type
    # anime loss(real)
    # anime gray loss(gray)
    # fake loss(real_d)
    # smooth loss(anime_smooth_d)
    anime_loss = 0
    anime_gray_loss = 0
    fake_loss = 0
    smooth_loss = 0

    bce_loss = nn.BCELoss()

    # loss func type
    # (wgan-gp, wgan-lp), lsgan, dragan, hinge
    if loss_func_type in ["wgan-gp", "wgan-lp"]:
        anime_loss = - torch.mean(anime_d)
        gray_loss = torch.mean(anime_gray_d)
        fake_loss = torch.mean(real_d)
        smooth_loss = torch.mean(anime_smooth_d)

    if loss_func_type == 'lsgan':
        anime_loss = torch.mean(torch.square(anime_d - 1.0))
        gray_loss = torch.mean(torch.square(anime_gray_d))
        fake_loss = torch.mean(torch.square(real_d))
        smooth_loss = torch.mean(torch.square(anime_smooth_d))

    if loss_func_type in ["gan", "dragan"]:
        # ToDo: check shape
        anime_loss = torch.mean(bce_loss(anime_d, torch.ones_like(anime_d)))
        gray_loss = torch.mean(bce_loss(anime_gray_d, torch.zeros_like(anime_gray_d)))
        fake_loss = torch.mean(bce_loss(real_d, torch.zeros_like(real_d)))
        smooth_loss = torch.mean(bce_loss(anime_smooth_d, torch.zeros_like(anime_smooth_d)))

    if loss_func_type == 'hinge':
        anime_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - anime_d))
        gray_loss = torch.mean(nn.ReLU(inplace=True)(1.0 + anime_gray_d))
        fake_loss = torch.mean(nn.ReLU(inplace=True)(1.0 + real_d))
        smooth_loss = torch.mean(nn.ReLU(inplace=True)(1.0 + anime_smooth_d))

    # why?
    loss = 1.7 * anime_loss + 1.7 * fake_loss + 1.7 * gray_loss + 0.8 * smooth_loss

    return loss



