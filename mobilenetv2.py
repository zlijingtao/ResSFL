'''MobileNetV2 in PyTorch.
Fetched from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.


'''
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from architectures_torch import ResBlock

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class MobileNet(nn.Module):
    

    def __init__(self, feature, logger, num_client = 1, num_class = 10, initialize_different = False):
        super(MobileNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.current_client = 0

        self.local_list = []
        for i in range(num_client):
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

        # self.layers = self._make_layers(in_planes=32)

        self.local = self.local_list[0]
        self.cloud = feature[1]
        self.logger = logger
        self.initialize = True

        self.classifier = nn.Linear(1280, num_class)
        
        print("local:")
        print(self.local)
        print("cloud:")
        print(self.cloud)
        print("classifier:")
        print(self.classifier)

    def switch_model(self, client_id):
        self.current_client = client_id
        self.local = self.local_list[client_id]
    
    def get_current_client(self):
        return self.current_client

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
        local_output = self.local(x)
        x = self.cloud(local_output)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

def make_layers(cutting_layer, cfg, in_planes, adds_bottleneck = False, bottleneck_option = "C8S1"):
        local_layer_list = []
        cloud_layer_list = []
        current_layer = 0
        in_channels = 3
        if cutting_layer > current_layer:
            local_layer_list.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            local_layer_list.append(nn.BatchNorm2d(32))
            in_channels = 32
        else:
            cloud_layer_list.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            cloud_layer_list.append(nn.BatchNorm2d(32))
        
        for expansion, out_planes, num_blocks, stride in cfg:
            current_layer += 1
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if cutting_layer > current_layer:
                    local_layer_list.append(Block(in_planes, out_planes, expansion, stride))
                    in_channels = out_planes
                else:
                    cloud_layer_list.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        current_layer += 1
        if cutting_layer > current_layer:
            local_layer_list.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            local_layer_list.append(nn.BatchNorm2d(1280))
        else:
            cloud_layer_list.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            cloud_layer_list.append(nn.BatchNorm2d(1280))

        local = []
        cloud = []
        if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
            print("original channel size of smashed-data is {}".format(in_channels))
            try:
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
            # cleint-side bottleneck
            if bottleneck_stride == 1:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
            elif bottleneck_stride >= 2:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                for _ in range(int(np.log2(bottleneck_stride//2))):
                    local += [nn.ReLU()]
                    local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
            local += [nn.ReLU()]
            # server-side bottleneck
            if bottleneck_stride == 1:
                cloud += [nn.Conv2d(bottleneck_channel_size, in_channels, kernel_size=1, stride= 1)]
            elif bottleneck_stride >= 2:
                for _ in range(int(np.log2(bottleneck_stride//2))):
                    cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                    cloud += [nn.ReLU()]
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, in_channels, kernel_size=3, output_padding=1, padding=1, stride= 2)]
            cloud += [nn.ReLU()]
            print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        local_layer_list += local
        cloud_layer_list = cloud + cloud_layer_list

        return nn.Sequential(*local_layer_list), nn.Sequential(*cloud_layer_list)


def MobileNetV2(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return MobileNet(make_layers(cutting_layer,cfg, in_planes=32, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different)

# def test():
#     net = MobileNetV2(9, None)
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())

# test()
