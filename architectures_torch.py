# import tensorflow as tf
# import numpy as np
import functools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.linear import Linear

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


def resnet(input_shape, level):
    # print(level)
    net = []

    net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
    net += [nn.BatchNorm2d(64)]
    net += [nn.ReLU()]
    net += [nn.MaxPool2d(2)]
    net += [ResBlock(64, 64)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(64, 128, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock(128, 128)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock(128, 256, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)

def resnet_tail(level, num_class = 10):
    print(level)
    net = []
    if level <= 1:
        net += [ResBlock(64, 128, stride=2)]
    if level <= 2:
        net += [ResBlock(128, 128)]
    if level <= 3:
        net += [ResBlock(128, 256, stride=2)]
    net += [ResBlock(256, 256, stride=1)]
    net += [ResBlock(256, 512, stride=2)]
    net += [ResBlock(512, 512, stride=1)]
    net += [ResBlock(512, 1024, stride=2)]
    net += [ResBlock(1024, 1024, stride=1)]
    # net += [nn.AvgPool2d(2)]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(num_class)]
    return nn.Sequential(*net)


def pilot(input_shape, level):

    net = []

    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    
    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)
    
    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def make_generator(latent_size):

    net = []
    net += [torch.nn.Linear(latent_size, 8*8*256, bias = False)]
    net += [torch.nn.BatchNorm1d(8*8*256)]
    net += [torch.nn.LeakyReLU()]
    net += [View((-1, 256, 8, 8))]
    net += [torch.nn.ConvTranspose2d(256, 128, 3, 1, padding = 1, bias = False)]
    net += [torch.nn.BatchNorm2d(128)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(128, 64, 3, 2, padding = 1, output_padding=1, bias = False)]
    net += [torch.nn.BatchNorm2d(64)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(64, 3, 3, 2, padding = 1, output_padding=1, bias = False)]
    net += [torch.nn.Tanh()]
    # net += [torch.nn.Sigmoid()]

    return nn.Sequential(*net)


def multihead_buffer(feature_size):
    assert len(feature_size) == 4
    net = []
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    return nn.Sequential(*net)

def multihead_buffer_res(feature_size):
    assert len(feature_size) == 4
    net = []
    net += [ResBlock(feature_size[1], feature_size[1])]
    net += [ResBlock(feature_size[1], feature_size[1])]
    # net += [ResBlock(feature_size[1], feature_size[1])]
    return nn.Sequential(*net)

def cifar_pilot(output_dim, level):

    net = []

    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    print(output_dim)
    if output_dim[2] == 32:
        net += [nn.Conv2d(3, 64, 3, 1, 1)]
        return  nn.Sequential(*net)

    net += [nn.Conv2d(3, 64, 3, 2, 1)]

    net += [nn.Conv2d(64, 64, 3, 1, 1)]

    if output_dim[2] == 16:
        net += [nn.Conv2d(64, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)


    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    net += [nn.Conv2d(128, 128, 3, 1, 1)]

    if output_dim[2] == 8:
        net += [nn.Conv2d(128, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)
    
    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if output_dim[2] == 4:
        net += [nn.Conv2d(256, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)
        

def decoder(input_shape, level, channels=3):
    
    net = []

    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1)]

    if level == 1:
        net += [nn.Conv2d(256, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]

    if level <= 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(128, channels, 3, 2, 1, output_padding=1)]
    net += [nn.Tanh()]
    return nn.Sequential(*net)

def cifar_decoder(input_shape, channels=3):
    
    net = []

    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)
    # print(input_shape)

    if input_shape[2] == 16:
        net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    elif input_shape[2] == 8:
        net += [nn.Conv2d(input_shape[0], 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    elif input_shape[2] == 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    else:
        raise Exception('No Dim %d' % input_shape[2])

# def inference_model(input_shape):
#     pass

class inference_model(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(inference_model, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(num_classes,1024),
            # nn.Linear(num_classes,256),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            # nn.Linear(256,128),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,128),
            # nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,1024),
           nn.ReLU(),
            nn.Linear(1024,512),
        #    nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(512,128),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(128*2,256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        # for key in self.state_dict():
        #     if key.split('.')[-1] == 'weight':    
        #         nn.init.normal(self.state_dict()[key], std=0.01)
                
        #     elif key.split('.')[-1] == 'bias':
        #         self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,x,l):
        
        out_x = self.features(x)
        out_l = self.labels(l)
            
        is_member =self.combine( torch.cat((out_x  ,out_l),1))
    
        return self.output(is_member)


def discriminator(input_shape, level):

    net = []
    if level == 1:
        net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape[0], 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]
        
    bn = False
        
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(1)]
    return nn.Sequential(*net)
#==========================================================================================

class custom_AE_bn(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE_bn, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(int(nc/2))]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(int(nc/2))]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(int(input_nc/(2 ** (upsampling_num - 1))))]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(input_nc)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class custom_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            #TODO: change to Conv2d
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class conv_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(conv_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        model += [nn.Conv2d(input_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #first
        model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        for _ in range(N):
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #Middle-N
            model += [nn.BatchNorm2d(internal_nc)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [nn.Conv2d(internal_nc, output_nc, kernel_size=3, stride=1, padding=1)] #last
        model += [nn.BatchNorm2d(output_nc)]

        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output





class res_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(res_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
            
        model += [ResBlock(input_nc, internal_nc, bn = True, stride=1)] #first
        model += [nn.ReLU()]

        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)] #second
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [ResBlock(internal_nc, output_nc, bn = True, stride=1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output

def classifier_binary(input_shape, class_num):
    net = []
    # xin = tf.keras.layers.Input(input_shape)
    # net += [nn.ReLU()]
    # net += [nn.Conv2d(input_shape[0], 128, 3, 1, 1)]
    # net += [nn.ReLU()]
    # net += [ResBlock(128, 128, bn=True)]
    # net += [ResBlock(128, 128, bn=True)]
    net += [nn.ReLU()]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(256)]
    net += [nn.ReLU()]
    net += [nn.Linear(256, 128)]
    net += [nn.ReLU()]
    # if(class_num > 1):
    #     net += [nn.BatchNorm2d(np.prod([input_shape[0], 32, input_shape[2], input_shape[3]]))]
    net += [nn.Linear(128, class_num)]
    return nn.Sequential(*net)

def pilotClass(input_shape, level):
    net = []
    # xin = tf.keras.layers.Input(input_shape)

    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]
    net += [nn.SiLU]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]
    net += [nn.SiLU]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    net += [nn.SiLU]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)
        
SETUPS = [(functools.partial(resnet, level=i), functools.partial(pilot, level=i), functools.partial(decoder, level=i), functools.partial(discriminator, level=i), functools.partial(resnet_tail, level=i)) for i in range(1,6)]

# bin class
l = 4
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l), functools.partial(resnet_tail, level=l))]

l = 3
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l), functools.partial(resnet_tail, level=l))]