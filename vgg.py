'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
import torch.nn.functional as F
from thop import profile
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, feature, logger, num_agent = 1, num_class = 10, initialize_different = False):
        super(VGG, self).__init__()
        self.current_agent = 0
        
        self.local_list = []
        for i in range(num_agent):
            if i == 0:
                self.local_list.append(feature[0])
                self.local_list[0].apply(init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])

                self.local_list.append(new_copy.cuda())
                if initialize_different:
                    self.local_list[i].apply(init_weights)
                    
            # for name, params in self.local_list[-1].named_parameters():
            #     print(name, 'of client', i, params.data[1][1])
            #     break
        self.local = self.local_list[0]
        self.cloud = feature[1]
        self.logger = logger
        classifier_list = [nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)]
        classifier_list += [nn.Linear(512, num_class)]
        self.classifier = nn.Sequential(*classifier_list)

        print("local:")
        print(self.local)
        print("cloud:")
        print(self.cloud)
        print("classifier:")
        print(self.classifier)

    def switch_model(self, agent_id):
        self.current_agent = agent_id
        self.local = self.local_list[agent_id]

    def get_current_agent(self):
        return self.current_agent

    def get_smashed_data_size(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 32, 32])
            try:
                device = next(self.local.parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.local(noise_input)
        return smashed_data.size()

    def get_MAC_param(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 32, 32])
            device = next(self.local.parameters()).device
            noise_input = noise_input.to(device)
            client_macs, client_params = profile(self.local, inputs=(noise_input, ))
            noise_smash = torch.randn(self.get_smashed_data_size())
            device = next(self.cloud.parameters()).device
            noise_smash = noise_smash.to(device)
            server_macs, server_params = profile(self.cloud, inputs=(noise_smash, ))
            noise_final = self.cloud(noise_smash)
            noise_final = noise_final.view(noise_final.size(0), -1)
            clas_macs, clas_params = profile(self.classifier,inputs=(noise_final,))
            server_macs += clas_macs
            server_params += clas_params
        return client_macs, client_params, server_macs, server_params

    def get_inference_time(self, federated = False):
      import time
      with torch.no_grad():
          noise_input = torch.randn([128, 3, 32, 32])
          
          if not federated:
            #CPU warm up
            self.local.cpu()
            self.local.eval()
            smashed_data = self.local(noise_input) #CPU warm up
            
            start_time = time.time()
            for _ in range(500):
                smashed_data = self.local(noise_input)
            lapse_cpu = (time.time() - start_time)/500
          else:
            self.local.cpu()
            self.cloud.cpu()
            self.classifier.cpu()
            self.local.eval()
            self.cloud.eval()
            self.classifier.eval()

            smashed_data = self.local(noise_input) #CPU warm up
            output = self.cloud(smashed_data)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
            start_time = time.time()
            for _ in range(500):
                output = self.local(noise_input)
                output = self.cloud(output)
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
            lapse_cpu = (time.time() - start_time)/500
          
          if not federated:
            self.local.cuda()
            smashed_data = smashed_data.cuda()
            self.cloud.eval()
            #GPU-WARM-UP
            for _ in range(100):  #GPU warm up
                output = self.cloud(smashed_data)
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
            start_time = time.time()
            for _ in range(500):
                output = self.cloud(smashed_data)
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
            lapse_gpu = (time.time() - start_time)/500
          else:
            self.local.cuda()
            self.cloud.cuda()
            self.classifier.cuda()
            lapse_gpu = 0.0
          del noise_input, output, smashed_data
      return lapse_cpu, lapse_gpu

    def get_train_time(self, federated = False):
        import time
        noise_input = torch.randn([128, 3, 32, 32])
        noise_label = torch.randint(0, 10, [128, ])
        self.local.cpu()
        self.cloud.cpu()
        self.classifier.cpu()
        self.local.train()
        self.cloud.train()
        self.classifier.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        
        '''Calculate client backward on CPU'''
        smashed_data = self.local(noise_input) #CPU warm up
        output = self.cloud(smashed_data)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        f_loss = criterion(output, noise_label)

        f_loss.backward()

        lapse_cpu_all = 0
        for _ in range(500):
            smashed_data = self.local(noise_input)
            output = self.cloud(smashed_data)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
            f_loss = criterion(output, noise_label)
            start_time = time.time()
            f_loss.backward()
            #First time we calculate CPU overall train time.
            lapse_cpu_all += (time.time() - start_time)
        lapse_cpu_all = lapse_cpu_all / 500.

        if not federated:
            lapse_cpu_server = 0
            for _ in range(500):
                smashed_data = self.local(noise_input)
                output = self.cloud(smashed_data.detach())
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
                f_loss = criterion(output, noise_label)

                start_time = time.time()
                f_loss.backward()
                #First time we calculate CPU server train time by detaching smashed-data.
                lapse_cpu_server += (time.time() - start_time)
            lapse_cpu_server = lapse_cpu_server / 500.

            lapse_cpu_client = lapse_cpu_all - lapse_cpu_server
        else: # if federated
            lapse_cpu_client = lapse_cpu_all
        
        '''Calculate Server backward on GPU'''
        self.local.cuda()
        self.cloud.cuda()
        self.classifier.cuda()
        if not federated:
            criterion.cuda()
            noise_input = noise_input.cuda()
            noise_label = noise_label.cuda()
            
            #GPU warmup
            for _ in range(100):
                smashed_data = self.local(noise_input)
                output = self.cloud(smashed_data.detach())
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
                f_loss = criterion(output, noise_label)
                f_loss.backward()

            lapse_gpu_server = 0
            for _ in range(500):
                smashed_data = self.local(noise_input)
                output = self.cloud(smashed_data.detach())
                output = output.view(output.size(0), -1)
                output = self.classifier(output)
                f_loss = criterion(output, noise_label)

                start_time = time.time()
                f_loss.backward()
                #First time we calculate CPU server train time by detaching smashed-data.
                lapse_gpu_server += (time.time() - start_time)
            lapse_gpu_server = lapse_gpu_server / 500.
        else:
            lapse_gpu_server = 0.0
        return lapse_cpu_client, lapse_gpu_server

    def forward(self, x):
        self.local_output = self.local(x)
        x = self.cloud(self.local_output)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_vib(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, feature, logger, num_agent = 1, num_class = 10, initialize_different = False):
        super(VGG_vib, self).__init__()
        self.current_agent = 0
        
        self.local_list = []
        for i in range(num_agent):
            if i == 0:
                self.local_list.append(feature[0])
                self.local_list[0].apply(init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])

                self.local_list.append(new_copy.cuda())
                if initialize_different:
                    self.local_list[i].apply(init_weights)
                    
            for name, params in self.local_list[-1].named_parameters():
                print(name, 'of client', i, params.data[1][1])
                break
        self.local = self.local_list[0]
        self.cloud = feature[1]
        self.logger = logger
        classifier_list = [nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)]
        self.classifier = nn.Sequential(*classifier_list)
        
        self.feat_dim = np.prod(self.get_smashed_data_size()[1:])
        # print(self.feat_dim)
        # vib
        # self.feat_dim = 512
        self.k = self.feat_dim
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        # self.fc_layer = nn.Linear(self.k, num_class)

    def switch_model(self, agent_id):
        self.current_agent = agent_id
        self.local = self.local_list[agent_id]

    def get_current_agent(self):
        return self.current_agent

    def get_smashed_data_size(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 32, 32])
            device = next(self.local.parameters()).device
            noise_input = noise_input.to(device)
            smashed_data = self.local(noise_input)
        return smashed_data.size()

    def get_MAC_param(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 32, 32])
            client_macs, client_params = profile(self.local, inputs=(noise_input, ))
        with torch.no_grad():
            noise_smash = torch.randn(self.get_smashed_data_size())
            server_macs, server_params = profile(self.cloud, inputs=(noise_smash, ))
        return client_macs, client_params, server_macs, server_params

    def get_inference_time(self):
      import time
      with torch.no_grad():
          noise_input = torch.randn([128, 3, 32, 32])
          self.local.cpu()
          self.local.eval()
          #CPU warm up
          smashed_data = self.local(noise_input) #CPU warm up
          start_time = time.time()
          for _ in range(100):
              smashed_data = self.local(noise_input)
          lapse_cpu = (time.time() - start_time)/100
          self.local.cuda()
          smashed_data = smashed_data.cuda()
          self.cloud.eval()
          #GPU-WARM-UP
          for _ in range(20):  #GPU warm up
              output = self.cloud(smashed_data)
          start_time = time.time()
          for _ in range(100):
              output = self.cloud(smashed_data)
          lapse_gpu = (time.time() - start_time)/100
          del noise_input, output, smashed_data
      return lapse_cpu, lapse_gpu
    # def forward(self, x):
    #     self.local_output = self.local(x)

    #     feature = self.cloud(self.local_output)
    #     feature = self.classifier(feature)
    #     feature = feature.view(feature.size(0), -1)

    #     statis = self.st_layer(feature)
    #     mu, std = statis[:, :self.k], statis[:, self.k:]

    #     std = F.softplus(std-5, beta=1)
    #     eps = torch.FloatTensor(std.size()).normal_().cuda()
    #     res = mu + std * eps


    #     out = self.fc_layer(res)
    #     return [feature, mu, std, out]

    def forward(self, x):
        feature = self.local(x)
        feature_old_size = feature.size()
        feature = feature.view(feature.size(0), -1)
        # print(feature.size())
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
    
        res = res.view(feature_old_size)
        x = self.cloud(res)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [feature, mu, std, x]

def make_layers(cutting_layer,cfg, batch_norm=False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    local = []
    cloud = []
    in_channels = 3
    
    #Modified Local part - Experimental feature
    channel_mul = 1
    for v_idx,v in enumerate(cfg):
        if v_idx < cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1)
                if batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = int(v * channel_mul)
        elif v_idx == cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
                print("original channel size of smashed-data is {}".format(in_channels))
                try:
                    if "noRELU" in bottleneck_option or "norelu" in bottleneck_option or "noReLU" in bottleneck_option:
                        relu_option = False
                    else:
                        relu_option = True
                    if "K" in bottleneck_option:
                        bn_kernel_size = int(bottleneck_option.split("C")[0].split("K")[1])
                    else:
                        bn_kernel_size = 3
                    bottleneck_channel_size = int(bottleneck_option.split("S")[0].split("C")[1])
                    if "S" in bottleneck_option:
                        bottleneck_stride = int(bottleneck_option.split("S")[1])
                    else:
                        bottleneck_stride = 1
                except:
                    print("auto extract bottleneck option fail (format: CxSy, x = [1, max_channel], y = {1, 2}), set channel size to 8 and stride to 1")
                    bn_kernel_size = 3
                    bottleneck_channel_size = 8
                    bottleneck_stride = 1
                    relu_option = True
                # cleint-side bottleneck
                if bottleneck_stride == 1:
                    local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
                elif bottleneck_stride >= 2:
                    local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                    for _ in range(int(np.log2(bottleneck_stride//2))):
                        if relu_option:
                            local += [nn.ReLU()]
                        local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                if relu_option:
                    local += [nn.ReLU()]
                # server-side bottleneck
                if bottleneck_stride == 1:
                    cloud += [nn.Conv2d(bottleneck_channel_size, in_channels, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
                elif bottleneck_stride >= 2:
                    for _ in range(int(np.log2(bottleneck_stride//2))):
                        cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                        if relu_option:
                            cloud += [nn.ReLU()]
                    cloud += [nn.ConvTranspose2d(bottleneck_channel_size, in_channels, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                if relu_option:
                    cloud += [nn.ReLU()]
                print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        else:
            if v == 'M':
                cloud += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    cloud += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    cloud += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

    return nn.Sequential(*local), nn.Sequential(*cloud)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cutting_layer,cfg['A'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg11_bn(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['A'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg13(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cutting_layer,cfg['B'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg13_bn(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['B'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)

def vgg11_vib(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 11-layer model (configuration "A")"""
    return VGG_vib(make_layers(cutting_layer,cfg['A'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg11_bn_vib(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_vib(make_layers(cutting_layer,cfg['A'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg13_vib(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 13-layer model (configuration "B")"""
    return VGG_vib(make_layers(cutting_layer,cfg['B'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg13_bn_vib(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG_vib(make_layers(cutting_layer,cfg['B'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg16(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cutting_layer,cfg['D'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg16_bn(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['D'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg19(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cutting_layer,cfg['E'], batch_norm=False, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)


def vgg19_bn(cutting_layer, logger, num_agent = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['E'], batch_norm=True, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_agent = num_agent, num_class = num_class, initialize_different = initialize_different)