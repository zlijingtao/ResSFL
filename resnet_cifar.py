import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import copy
from thop import profile
import numpy as np
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      init.kaiming_normal(m.weight)
      m.bias.data.zero_()

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)

    self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + basicblock, inplace=True)

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.current_client = 0
    

    layers = []
    layers.append(conv3x3(3, 16))
    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
    layers.extend(self.stage_1)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
    layers.extend(self.stage_2)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
    layers.extend(self.stage_3)
    # local_layer_list = layers[:cutting_layer]
    # cloud_layer_list = layers[cutting_layer:]
    try:
        local_layer_list = layers[:cutting_layer]
        cloud_layer_list = layers[cutting_layer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []
    
    temp_local = nn.Sequential(*local_layer_list)
    with torch.no_grad():
        noise_input = torch.randn([1, 3, 32, 32])
        smashed_data = temp_local(noise_input)
    in_channels = smashed_data.size(1)
    
    print("in_channels is {}".format(in_channels))

    local = []
    cloud = []
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
    local_layer_list += local
    cloud_layer_list = cloud + cloud_layer_list
    self.local = nn.Sequential(*local_layer_list)
    self.cloud = nn.Sequential(*cloud_layer_list)
    self.local_list = []
    for i in range(num_client):
        if i == 0:
            self.local_list.append(self.local)
            self.local_list[0].apply(init_weights)
        else:
            new_copy = copy.deepcopy(self.local_list[0])

            self.local_list.append(new_copy.cuda())
            if initialize_different:
                self.local_list[i].apply(init_weights)
                
        # for name, params in self.local_list[-1].named_parameters():
        #     print(name, 'of client', i, params.data[1][1])
        #     break
    self.logger = logger
    self.classifier = nn.Linear(64*block.expansion, num_class)
    print("local:")
    print(self.local)
    print("cloud:")
    print(self.cloud)
    print("classifier:")
    print(self.classifier)
    for m in self.cloud:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
  def switch_model(self, client_id):
        self.current_client = client_id
        self.local = self.local_list[client_id]

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
            noise_final = F.avg_pool2d(noise_final, 8)
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
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
            start_time = time.time()
            for _ in range(500):
                smashed_data = self.local(noise_input)
                output = self.cloud(smashed_data)
                output = F.avg_pool2d(output, 8)
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
            start_time = time.time()
            for _ in range(500):
                output = self.cloud(smashed_data)
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
        output = F.avg_pool2d(output, 8)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        f_loss = criterion(output, noise_label)

        f_loss.backward()

        lapse_cpu_all = 0
        for _ in range(500):
            smashed_data = self.local(noise_input)
            output = self.cloud(smashed_data)
            output = F.avg_pool2d(output, 8)
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
              output = F.avg_pool2d(output, 8)
              output = output.view(output.size(0), -1)
              output = self.classifier(output)
              f_loss = criterion(output, noise_label)

              start_time = time.time()
              f_loss.backward()
              #First time we calculate CPU server train time by detaching smashed-data.
              lapse_cpu_server += (time.time() - start_time)
          lapse_cpu_server = lapse_cpu_server / 500.

          lapse_cpu_client = lapse_cpu_all - lapse_cpu_server
        else:
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
              output = F.avg_pool2d(output, 8)
              output = output.view(output.size(0), -1)
              output = self.classifier(output)
              f_loss = criterion(output, noise_label)
              f_loss.backward()

          lapse_gpu_server = 0
          for _ in range(500):
              smashed_data = self.local(noise_input)
              output = self.cloud(smashed_data.detach())
              output = F.avg_pool2d(output, 8)
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
  
  def _make_layer(self, block, planes, blocks, stride=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
          layers.append(block(self.inplanes, planes))
      return layers

  def forward(self, x):
      self.local_output = self.local(x)
      x = self.cloud(self.local_output)
      x = F.avg_pool2d(x, 8)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x

class conv3x3(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

def ResNet20(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
  """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 20, cutting_layer, logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option)
  return model

def ResNet32(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
  """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 32, cutting_layer, logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option)
  return model

def ResNet44(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
  """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 44, cutting_layer, logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option)
  return model

def ResNet56(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
  """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 56, cutting_layer, logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option)
  return model

def ResNet110(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
  """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNet(ResNetBasicblock, 110, cutting_layer, logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option)
  return model

# def test():
#     net = ResNet20(1, None)
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()