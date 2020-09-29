import numpy as np
import torch
import torch.nn as nn
# build optimizer
from torch.optim import Adam
from dataset import data_load
from torchvision import transforms

from subnetwork import VGG19, Generator, Discriminator

# Network architecture => VGG19 will be used, so need to prepare for pretrained VGG19 model

# real, anime, anime_smooth, anime_gray

# input => (batch_size, channel, image_size, image_size)
# output => (batch_size, )

# Point
# like CartoonGAN, image space converts rgb into yuv


class AnimeGAN(nn.Module):

    def __init__(self, args, dataset, phase, device):


        super().__init__()

        self.phase = phase
        self.device = device
        self.epoch = args.epoch
        self.save_freq = args.save_freq

        self.generator = Generator()
        self.discriminator = Discriminator()

        if args.resume_g_path != "":
            generator_state_dict_path = args.resume_g_path
            names = generator_state_dict_path.split('/')
            resume_epoch = names[-1].split('.')[0].split('_')[-1]
            names[-1] = 'checkpoint_discriminator_{}.pth.tar'.format(resume_epoch)
            discriminator_state_dict_path = ''
            for n in names:
                discriminator_state_dict_path += '/{}'.format(n)
            
            self.generator.load_state_dict(torch.load(generator_state_dict_path))
            self.discriminator.load_state_dict(torch.load(discriminator_state_dict_path))

        # to cuda
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        self.vgg19 = VGG19() # pretrained VGG19
        self.vgg19.to(device)

        self.loss_func_type = args.loss_func_type

        self.con_weight = args.con_weight
        self.style_weight = args.style_weight
        self.color_weight = args.color_weight

        self.g_optimizer = Adam(self.generator.parameters(), lr=args.lr_g, beta1=args.beta1, beta2=args.beta2)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=args.lr_d, beta1=args.beta1, beta2=args.beta2)

        # transform
        real_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        anime_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        anime_smooth_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        anime_gray_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # create dataloader class
        real_image_dataset_path = args.dataset_dir + 'real/'
        anime_image_dataset_path = args.dataset_dir + '{}/anime/'.format(args.anime_name)
        anime_smooth_dataset_path = args.dataset_dir + '{}/anime_smooth/'.format(args.anime_name)
        anime_gray_dataset_path = args.dataset_dir + '{}/anime_gray/'.format(args.anime_name)
        self.real_image_dataloader = data_load(real_image_dataset_path, real_transform, args.batch_size)
        self.anime_image_dataloader = data_load(anime_image_dataset_path, anime_transform, args.batch_size)
        self.anime_smooth_dataloader = data_load(anime_smooth_dataset_path, anime_smooth_transform, args.batch_size)
        self.anime_gray_dataloader = data_load(anime_gray_dataset_path, anime_gray_transform, args.batch_size)
        

    def model_parameters(self):

        return list(self.generator.parameters()) + list(self.discriminator.parameters())

    def build_model(self, real_image, anime_image, anime_smooth, anime_gray):
        # calculate loss

        # generator inference
        real_g = self.generator(real_image)  # G(x), x = real image

        # disctiminator inference
        real_d = self.discriminator(real_g)  # D(G(x))
        anime_d = self.discriminator(anime_image)  # D(y)
        anime_gray_d = self.discriminator(anime_gray)  # D(y_gray)
        anime_smooth_d = self.discriminator(anime_smooth)  # D(y_smoo)

        # add gradient penalty(Drastic GAN?)
        GP = 0.0

        # ToDo: implement con_sty_loss(), color_loss(), generator_loss(), discriminator_loss()
        # calculate con style loss(c_loss => con loss, s_loss => style loss)
        c_loss, s_loss = con_sty_loss(self.vgg19, real_image, anime_gray_d, real_g)
        t_loss = self.con_weight * c_loss + self.style_weight * s_loss + self.color_weight * color_loss(real_image, real_g)

        # generator loss and disctiminator loss
        g_loss = generator_loss(self.loss_func_type, real_d) * self.g_adv_weight
        d_loss = discriminator_loss(self.loss_func_type, anime_d, anime_gray_d, real_d, anime_smooth_d) * self.d_adv_weight

        generator_loss = g_loss + t_loss
        discriminator_loss = d_loss

        return generator_loss, discriminator_loss

    def pretrained_generator(self):
        # set mode
        self.generator.train()
        self.vgg19.eval()
        for e in range(self.pretrained_epoch):
            reconstruct_losses = []

            for (real_image, _), (anime_image, _), (anime_smooth, _), (anime_gray, _) in zip(self.real_image_dataloader, self.anime_image_dataloader, self.anime_smooth_dataloader, self.anime_gray_dataloader):
                # for cuda
                real_image = real_image.to(device)
                anime_image = anime_image.to(device)
                anime_smooth = anime_smooth.to(device)
                anime_gray = anime_gray.to(device)

                # train generator
                generator_loss, _ = self.build_model(real_image, anime_image, anime_smooth, anime_gray)

                self.g_optimizer.zero_grad()
                generator_loss.backward()
                self.g_optimizer.step()

                reconstruct_losses.append(generator_loss.item())

            mean_recon_loss = np.mean(reconstruct_losses)

            print('pretrained phase epoch: {} generator loss: {:. 5f}'.format(e, mean_recon_loss))




    def train(self):
        # CartoonGAn, AnimeGAN
        # learning style
        # 1. pretrained generator training
        # 2. generator + discriminator training

        # pretrain generator
        self.pretrained_generator()

        # set mode
        self.generator.train()
        self.discriminator.train()
        self.vgg19.eval()
        
        for e in range(self.epoch):

            g_losses = []
            d_losses = []

            for (real_image, _), (anime_image, _), (anime_smooth, _), (anime_gray, _) in zip(self.real_image_dataloader, self.anime_image_dataloader, self.anime_smooth_dataloader, self.anime_gray_dataloader):
                # for cuda
                real_image = real_image.to(device)
                anime_image = anime_image.to(device)
                anime_smooth = anime_smooth.to(device)
                anime_gray = anime_gray.to(device)

                # calculate generator loss and discriminator loss
                generator_loss, discriminator_loss = self.build_model(real_image, anime_image, anime_smooth, anime_gray)
                # back propagation
                if e % self.trainig_rate == 0:
                    # update D
                    self.d_optimizer.zero_grad()
                    discriminator_loss.backward()
                    self.d_optimizer.step()
                
                # update G
                self.g_optimizer.zero_grad()
                generator_loss.backward()
                self.g_optimizer.step()

                g_losses.append(generator_loss.item())
                d_losses.append(discriminator_loss.item())

            mean_g_loss = np.mean(g_losses)
            mean_d_loss = np.mean(d_losses)

            print('epoch: {}  generator loss: {:.5f}  discriminator loss: {:.5f}'.format(e, mean_g_loss, mean_d_loss))

            if e % self.save_freq == 0:
                # generator
                generator_weight_path = 'checkpoint_generator_%05d.pth.tar' % e
                self.save(generator_weight_path, e, self.generator)

                # discriminator
                discriminator_weight_path = 'checkpoint_discriminator_%05d.pth.tar' % e
                self.save(discriminator_weight_path, e, self.discriminator)


                

    def predict(self, real_image):
        self.generator.eval()

        return self.generator(real_image)


    def save(self, weight_path, epoch, model):
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
        }, weight_path)


    

    




