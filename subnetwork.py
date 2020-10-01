import torch
import torch.nn as nn
"""
# VGG19
class VGG19(nn.Module):

    def __init__(self, pretrained_weights=None, feature_mode=True):

        super().__init__()

        self.feature_mode = feature_mode

        self.features = self.make_layers()

        if pretrained_weights is not None:

            self.load_state_dict(torch.load(pretrained_weights))
    
    def make_layers(self):
        # Conv(ch=64) * 2, MaxPooling2d(k=2, s=2), Conv(ch=128) * 2, MaxPooling2d(k=2, s=2), Conv(ch=256) * 4, MaxPooling2d(k=2, s=2),Conv(ch=512) * 4, MaxPooling2d(k=2, s=2), Conv(ch=512) * 4
        # classifier => MaxPool2d(), fc(4096), fc(4096), fc(1000), softmax()

        # input size => (3, 128, 128)  rgb image

        in_channels = 3
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 128, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 128, 128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 64, 64, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (batch_size, 128, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (batch_size, 128, 64, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 128, 32, 32)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, 32, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 256, 16, 16)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 512, 8, 8)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (batch_size, 512, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 512, 4, 4)

            nn.Linear(in_features=512 * 4 * 4, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1024),

        ]



        return nn.Sequential(*layers)

    def forward(self, x):

        # feature map => conv block 4's final layer output
        if self.feature_mode:
            layers_list = list(self.features.modules())

            for l in layers_list[1:27]:
                x = l(x)

        # if you do not need feature map, you add the classifier layer
        if not self.feature_mode:
            layers_list = list(self.features.modules())
            conv_layer_list = layers_list[:38]
            fc_layer_list = layers_list[38:]

            for cl in conv_layer_list:
                x = cl(x)

            x = x.view(x.size(0), -1)
            
            for fl in fc_layer_list:
                x = fl(x)



        return x
"""


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000, device='cpu'):
        super().__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.device = device
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        # print('vgg input shape:', x.shape)
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:
                x = l(x)
        
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)


        return x


class DepthWiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):

        super().__init__()

        # conv + instance norm + leaky Relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(num_features=out_channels)


    def forward(self, x):
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm(self.conv(x)))
        return x

class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        # (depthwise conv + instance norm + leaky Relu) + conv block
        self.depthwiseconv = DepthWiseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(num_features=out_channels)

        self.convblock = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm(self.depthwiseconv(x)))

        x = self.convblock(x)

        return x

class InvertedResblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # conv block + depthwise conv + instance norm + leaky Relu) + (conv + instance norm)
        self.convblock = ConvBlock(in_channels=in_channels, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.depthwiseconv = DepthWiseConv(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.instance_norm1 = nn.InstanceNorm2d(num_features=512)

        self.conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(num_features=256)

    def forward(self, x):
        x1 = x
        x = self.convblock(x)
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm1(self.depthwiseconv(x)))
        x = self.instance_norm2(self.conv(x))
        # print()
        # print('inversed Residual Block')
        # print('input:', x1.shape, 'output:', x.shape)
        x = x + x1  # residual convolution
        return x

class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dsconv1 = DSConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.resize = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.dsconv2 = DSConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        x1 = x
        x1 = self.resize(x1)  # (h / 2) * (w / 2)
        x1 = self.dsconv2(x1)

        x = self.dsconv1(x)
        # print()
        # print('DSConv')
        # print('input:', x1.shape, 'output:', x.shape)
        x = x + x1
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.resize = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.dsconv = DSConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.resize(x)  # 2h * 2w
        x = self.dsconv(x)  # 2h * 2w

        return x



# GAN
class Generator(nn.Module):
    """
    NetWotk architecture

    input

    conv block(channels=64)
    conv block(channels=64)
    down conv(channels=128)
    conv block(channels=128)
    ds conv(channels=128)
    down conv(channels=256)
    conv block(channels=256)

    IRB(in_channels=512, out_channels=256) * 8
    conv block(channels=256)

    up conv(channels=128)
    ds conv(channels=128)
    conv block(channels=128)

    up conv(channels=64)
    conv block(channels=64)
    conv block(channels=64)

    conv(channels=3)

    output

    """


    def __init__(self, in_channels=3, out_channels=3):

        super().__init__()

        # input => (batch_size, 3, 256, 256)
        self.convblock1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 256, 256)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 256, 256)
        self.downconv1 = DownConv(in_channels=64, out_channels=128)  # (128, 128, 128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 128, 128)
        self.dsconv1 = DSConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 128, 128)
        self.downconv2 = DownConv(in_channels=128, out_channels=256)  # (256, 64, 64)
        self.convblock4 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 64, 64)

        self.irb1 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb2 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb3 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb4 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb5 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb6 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb7 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)
        self.irb8 = InvertedResblock(in_channels=256, out_channels=256)  # (256, 64, 64)

        self.convblock5 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 64, 64)
        self.upconv1 = UpConv(in_channels=256, out_channels=128)  # (128, 128, 128)
        self.dsconv2 = DSConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 128, 128)
        self.convblock6 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 128, 128)
        self.upconv2 = UpConv(in_channels=128, out_channels=64)  # (64, 256, 256)
        self.convblock7 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 256, 256)
        self.convblock8 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 256, 256)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (3, 256, 256)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.downconv1(x)
        x = self.convblock3(x)
        x = self.dsconv1(x)
        x = self.downconv2(x)
        x = self.convblock4(x)

        x = self.irb1(x)
        x = self.irb2(x)
        x = self.irb3(x)
        x = self.irb4(x)
        x = self.irb5(x)
        x = self.irb6(x)
        x = self.irb7(x)
        x = self.irb8(x)

        x = self.convblock5(x)
        x = self.upconv1(x)
        x = self.dsconv2(x)
        x = self.convblock6(x)
        x = self.upconv2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        x = nn.Tanh()(self.final_conv(x))

        return x



class Discriminator(nn.Module):
    """
    Discriminator Network Architecture
    
    input  (3, 256, 256)

    conv (channels=32, k=3, s=1)
    LeakyReLU()

    conv (channels=64, k=3, s=2)
    LeakyReLU()

    conv (channels=128, k=3, s=1)
    instance norm()
    LeakyReLU()


    conv (channels=128, k=3, s=2)
    LeakyReLU()
    conv (channels=256, k=3, s=1)
    instance norm()
    LeakyReLU()


    conv (channels=256, k=3, s=1)
    instance norm()
    LeakyReLU()


    conv (channels=1, k=3, s=1)
    
    output (1, , )



    """

    def __init__(self, in_channels=3, out_channels=1):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 256, 256)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # (64, 128, 128)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 128, 128)
        self.instance_norm3 = nn.InstanceNorm2d(num_features=128)  # (128, 128, 128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)  # (128, 64, 64)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 64, 64)
        self.instance_norm5 = nn.InstanceNorm2d(num_features=256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 64, 64)
        self.instance_norm6 = nn.InstanceNorm2d(num_features=256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (1, 64, 64)


    def forward(self, x):
        
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.conv1(x))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.conv2(x))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm3(self.conv3(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.conv4(x))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm5(self.conv5(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.instance_norm6(self.conv6(x)))
        x = self.conv7(x)
        # x = nn.Sigmoid()(self.conv7(x))

        return x
