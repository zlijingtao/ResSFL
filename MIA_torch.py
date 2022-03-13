import torch
import numpy as np
import torch.nn as nn
from torch.serialization import save
import architectures_torch as architectures
from utils import setup_logger, accuracy, AverageMeter, WarmUpLR, apply_transform_test, apply_transform, TV, l2loss, dist_corr, get_PSNR
from utils import freeze_model_bn, average_weights, DistanceCorrelationLoss, spurious_loss, prune_top_n_percent_left, dropout_defense, prune_defense
from thop import profile
import logging
from torch.autograd import Variable
from resnet_cifar import ResNet20, ResNet32
from mobilenetv2 import MobileNetV2
from vgg import vgg11, vgg13, vgg11_bn, vgg13_bn
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from datetime import datetime
import os
from shutil import rmtree
from datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_mnist_bothloader, get_facescrub_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "mnist" or dataset == "fmnist":
        return torch.clamp((x + 1)/2, 0, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = (tensor[t]).mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)

def test_denorm():
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar10_training = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    cifar10_training_loader_iter = iter(DataLoader(cifar10_training, shuffle=False, num_workers=1, batch_size=128))
    transform_orig = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar10_original = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_orig)
    cifar10_original_loader_iter = iter(DataLoader(cifar10_original, shuffle=False, num_workers=1, batch_size=128))
    images, _ = next(cifar10_training_loader_iter)
    orig_image, _  = next(cifar10_original_loader_iter)
    recovered_image = denormalize(images, "cifar100")
    return torch.isclose(orig_image, recovered_image)

def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64):
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for img_idx in range(output_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)

        if input_imgs is not None:
            save_image(input_imgs[img_idx], inp_img_path)
        if output_imgs is not None:
            save_image(output_imgs[img_idx], out_img_path)

class MIA:

    def __init__(self, arch, cutting_layer, batch_size, n_epochs, scheme="V2_epoch", num_client=2, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0,
                 collude_use_public=False, initialize_different=False, learning_rate=0.1, local_lr = -1,
                 gan_AE_type="custom", random_seed=123, client_sample_ratio = 1.0,
                 load_from_checkpoint = False, bottleneck_option="None", measure_option=False,
                 optimize_computation=1, decoder_sync = False, bhtsne_option = False, gan_loss_type = "SSIM", attack_confidence_score = False,
                 ssim_threshold = 0.0, finetune_freeze_bn = False, load_from_checkpoint_server = False, source_task = "cifar100", 
                 save_activation_tensor = False, save_more_checkpoints = False):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.bhtsne = bhtsne_option
        self.batch_size = batch_size
        self.lr = learning_rate
        self.finetune_freeze_bn = finetune_freeze_bn
        if local_lr == -1: # if local_lr is not set
            self.local_lr = self.lr
        else:
            self.local_lr = local_lr
        self.n_epochs = n_epochs
        self.measure_option = measure_option
        self.optimize_computation = optimize_computation
        self.client_sample_ratio = client_sample_ratio
        # setup save folder
        if save_dir is None:
            self.save_dir = "./saves/{}/".format(datetime.today().strftime('%m%d%H%M'))
        else:
            self.save_dir = str(save_dir) + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_more_checkpoints = save_more_checkpoints
        
        # setup tensorboard
        tensorboard_path = str(save_dir) + "/tensorboard"
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.writer = SummaryWriter(log_dir=tensorboard_path)
        self.save_activation_tensor = save_activation_tensor

        # setup logger
        model_log_file = self.save_dir + '/MIA.log'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger('{}_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
        self.warm = 1
        self.scheme = scheme

        # migrate old naming:
        if self.scheme == "V1" or self.scheme == "V2" or self.scheme == "V3" or self.scheme == "V4":
            self.scheme = self.scheme + "_batch"

        self.num_client = num_client
        self.dataset = dataset
        self.call_resume = False

        self.load_from_checkpoint = load_from_checkpoint
        self.load_from_checkpoint_server = load_from_checkpoint_server
        self.source_task = source_task
        self.cutting_layer = cutting_layer

        if self.cutting_layer == 0:
            self.logger.debug("Centralized Learning Scheme:")
        if "resnet20" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/10".format(self.cutting_layer))
        if "vgg11" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/13".format(self.cutting_layer))
        if "mobilenetv2" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/9".format(self.cutting_layer))
        
        self.confidence_score = attack_confidence_score
        
        self.collude_use_public = collude_use_public
        self.initialize_different = initialize_different
        
        if "C" in bottleneck_option or "S" in bottleneck_option:
            self.adds_bottleneck = True
            self.bottleneck_option = bottleneck_option
        else:
            self.adds_bottleneck = False
            self.bottleneck_option = bottleneck_option
        
        self.decoder_sync = decoder_sync
        # Activation Defense:
        self.regularization_option = regularization_option

        # If strength is 0.0, then there is no regularization applied, train normally.
        self.regularization_strength = regularization_strength
        if self.regularization_strength == 0.0:
            self.regularization_option = "None"

        # setup nopeek regularizer
        if "nopeek" in self.regularization_option:
            self.nopeek = True
        else:
            self.nopeek = False

        self.alpha1 = regularization_strength  # set to 0.1 # 1000 in Official NoteBook https://github.com/tremblerz/nopeek/blob/master/noPeekCifar10%20(1)-Copy2.ipynb

        # setup gan_adv regularizer
        self.gan_AE_activation = "sigmoid"
        self.gan_AE_type = gan_AE_type
        self.gan_loss_type = gan_loss_type
        self.gan_decay = 0.2
        self.alpha2 = regularization_strength  # set to 1~10
        self.pretrain_epoch = 100

        self.ssim_threshold = ssim_threshold
        if "gan_adv" in self.regularization_option:
            self.gan_regularizer = True
            if "step" in self.regularization_option:
                try:
                    self.gan_num_step = int(self.regularization_option.split("step")[-1])
                except:
                    print("Auto extract step fail, geting default value 3")
                    self.gan_num_step = 3
            else:
                self.gan_num_step = 3
            if "noise" in self.regularization_option:
                self.gan_noise = True
            else:
                self.gan_noise = False
        else:
            self.gan_regularizer = False
            self.gan_noise = False

        # setup local dp (noise-injection defense)
        if "local_dp" in self.regularization_option:
            self.local_DP = True
        else:
            self.local_DP = False

        self.dp_epsilon = regularization_strength

        if "spurious_activation" in self.regularization_option:
            self.spurious_activation = True
        else:
            self.spurious_activation = False

        self.alpha4 = regularization_strength  # set to 1~10

        if "dropout" in self.regularization_option:
            self.dropout_defense = True
            try: 
                self.dropout_ratio = float(self.regularization_option.split("dropout")[1].split("_")[0])
            except:
                self.dropout_ratio = regularization_strength
                print("Auto extract dropout ratio fail, use regularization_strength input as dropout ratio")
        else:
            self.dropout_defense = False
            self.dropout_ratio = regularization_strength
        
        if "topkprune" in self.regularization_option:
            self.topkprune = True
            try: 
                self.topkprune_ratio = float(self.regularization_option.split("topkprune")[1].split("_")[0])
            except:
                self.topkprune_ratio = regularization_strength
                print("Auto extract topkprune ratio fail, use regularization_strength input as topkprune ratio")
        else:
            self.topkprune = False
            self.topkprune_ratio = regularization_strength
        
        # dividing datasets to actual number of clients, self.num_clients is fake num of clients for ease of simulation.
        multiplier = 1/self.client_sample_ratio #100
        actual_num_users = int(multiplier * self.num_client)
        self.actual_num_users = actual_num_users

        # setup dataset
        if self.dataset == "cifar10":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar10_trainloader(batch_size=self.batch_size,
                                                                                                        num_workers=4,
                                                                                                        shuffle=True,
                                                                                                        num_client=num_client,
                                                                                                        collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar10_testloader(batch_size=self.batch_size,
                                                                                                        num_workers=4,
                                                                                                        shuffle=False)
            self.orig_class = 10
        elif self.dataset == "cifar100":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar100_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=True,
                                                                                                         num_client=num_client,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar100_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=False)
            self.orig_class = 100

        elif self.dataset == "svhn":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_SVHN_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=True,
                                                                                                         num_client=num_client,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_SVHN_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=False)
            self.orig_class = 10

        elif self.dataset == "facescrub":
            self.client_dataloader, self.pub_dataloader = get_facescrub_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=num_client,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 530
        elif self.dataset == "mnist":
            self.client_dataloader, self.pub_dataloader = get_mnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=num_client,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        elif self.dataset == "fmnist":
            self.client_dataloader, self.pub_dataloader = get_fmnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=num_client,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        else:
            raise ("Dataset {} is not supported!".format(self.dataset))
        self.num_class = self.orig_class


        self.num_batches = len(self.client_dataloader[0])
        print("Total number of batches per epoch for each client is ", self.num_batches)

        self.model = None

        if "V" in self.scheme:
            # V1, V2 initialize must be the same
            if "V1" in self.scheme or "V2" in self.scheme:
                self.initialize_different = False

            if arch == "resnet20":
                model = ResNet20(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet32":
                model = ResNet32(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13":
                model = vgg13(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11":
                model = vgg11(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13_bn":
                model = vgg13_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11_bn":
                model = vgg11_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "mobilenetv2":
                model = MobileNetV2(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            else:
                raise ("No such architecture!")
            self.model = model

            self.f = model.local_list[0]
            if self.num_client > 1:
                self.c = model.local_list[1]
            self.f_tail = model.cloud
            self.classifier = model.classifier
            self.f.cuda()
            self.f_tail.cuda()
            self.classifier.cuda()
            self.params = list(self.f_tail.parameters()) + list(self.classifier.parameters())
            self.local_params = []
            if cutting_layer > 0:
                self.local_params.append(self.f.parameters())
                for i in range(1, self.num_client):
                    self.model.local_list[i].cuda()
                    self.local_params.append(self.model.local_list[i].parameters())
        else:
            # If not V3, we set num_client to 1 when initializing the model, because there is only one version of local model.
            if arch == "resnet20":
                model = ResNet20(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet32":
                model = ResNet32(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13":
                model = vgg13(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11":
                model = vgg11(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13_bn":
                model = vgg13_bn(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11_bn":
                model = vgg11_bn(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "mobilenetv2":
                model = MobileNetV2(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            else:
                raise ("No such architecture!")
            self.model = model
            self.f = model.local
            self.c = self.f
            for i in range(1, self.num_client):
                self.model.local_list.append(self.f)
            self.f_tail = model.cloud
            self.classifier = model.classifier
            self.f.cuda()
            self.f_tail.cuda()
            self.classifier.cuda()
            self.params = list(self.f_tail.parameters()) + list(self.classifier.parameters())
            self.local_params = []
            if cutting_layer > 0:
                self.local_params.append(self.f.parameters())

        # setup optimizers
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        milestones = [60, 120, 160]
        if self.client_sample_ratio < 1.0:
            multiplier = 1/self.client_sample_ratio
            for i in range(len(milestones)):
                milestones[i] = int(milestones[i] * multiplier)
        self.local_optimizer_list = []
        self.train_local_scheduler_list = []
        self.warmup_local_scheduler_list = []
        for i in range(len(self.local_params)):
            self.local_optimizer_list.append(torch.optim.SGD(list(self.local_params[i]), lr=self.local_lr, momentum=0.9, weight_decay=5e-4))
            self.train_local_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.local_optimizer_list[i], milestones=milestones,
                                                                    gamma=0.2))  # learning rate decay
            self.warmup_local_scheduler_list.append(WarmUpLR(self.local_optimizer_list[i], self.num_batches * self.warm))

        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.num_batches * self.warm)
        
        # Set up GAN_ADV
        self.local_AE_list = []
        self.gan_params = []
        if self.gan_regularizer:
            feature_size = self.model.get_smashed_data_size()
            for i in range(self.num_client):
                if self.gan_AE_type == "custom":
                    self.local_AE_list.append(
                        architectures.custom_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                output_dim=32, activation=self.gan_AE_activation))
                elif "conv_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("conv_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from conv_normN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                elif "res_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("res_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from res_normN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                else:
                    raise ("No such GAN AE type.")
                self.gan_params.append(self.local_AE_list[i].parameters())
                self.local_AE_list[i].apply(init_weights)
                self.local_AE_list[i].cuda()
            self.gan_optimizer_list = []
            self.gan_scheduler_list = []
            milestones = [60, 120, 160]

            if self.client_sample_ratio < 1.0:
                multiplier = 1/self.client_sample_ratio
                for i in range(len(milestones)):
                    milestones[i] = int(milestones[i] * multiplier)
            for i in range(len(self.gan_params)):
                self.gan_optimizer_list.append(torch.optim.Adam(list(self.gan_params[i]), lr=1e-3))
                self.gan_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.gan_optimizer_list[i], milestones=milestones,
                                                                      gamma=self.gan_decay))  # learning rate decay
            
    def optimizer_step(self, set_client = False, client_id = 0):
        self.optimizer.step()
        if set_client and len(self.local_optimizer_list) > client_id:
            self.local_optimizer_list[client_id].step()
        else:
            for i in range(len(self.local_optimizer_list)):
                self.local_optimizer_list[i].step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].zero_grad()

    def scheduler_step(self, epoch = 0, warmup = False):
        if warmup:
            self.warmup_scheduler.step()
            for i in range(len(self.warmup_local_scheduler_list)):
                self.warmup_local_scheduler_list[i].step()
        else:
            self.train_scheduler.step(epoch)
            for i in range(len(self.train_local_scheduler_list)):
                self.train_local_scheduler_list[i].step(epoch)

    def gan_scheduler_step(self, epoch = 0):
        for i in range(len(self.gan_scheduler_list)):
            self.gan_scheduler_list[i].step(epoch)
    
    def train_target_step(self, x_private, label_private, client_id=0):
        self.f_tail.train()
        self.classifier.train()
        if "V" in self.scheme:
            self.model.local_list[client_id].train()
        else:
            self.f.train()
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Freeze batchnorm parameter of the client-side model.
        if self.load_from_checkpoint and self.finetune_freeze_bn:
            if client_id == 0:
                freeze_model_bn(self.f)
            elif client_id == 1:
                freeze_model_bn(self.c)
            else:
                freeze_model_bn(self.model.local_list[client_id])


        # Final Prediction Logits (complete forward pass)
        if client_id == 0:
            z_private = self.f(x_private)
        elif client_id == 1:
            z_private = self.c(x_private)
        else:
            z_private = self.model.local_list[client_id](x_private)

        if self.local_DP:
            if "laplace" in self.regularization_option:
                noise = torch.from_numpy(
                    np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=z_private.size())).cuda()
                z_private = z_private + noise.detach().float()
            else:  # apply gaussian noise
                delta = 10e-5
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                noise = sigma * torch.randn_like(z_private).cuda()
                z_private = z_private + noise.detach().float()
        if self.dropout_defense:
            z_private = dropout_defense(z_private, self.dropout_ratio)
        if self.topkprune:
            z_private = prune_defense(z_private, self.topkprune_ratio)
        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)  
            z_private = z_private - grad.detach() * epsilon

        output = self.f_tail(z_private)

        if "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if self.nopeek:
            #
            if "ttitcombe" in self.regularization_option:
                dc = DistanceCorrelationLoss()
                dist_corr_loss = self.alpha1 * dc(x_private, z_private)
            else:
                dist_corr_loss = self.alpha1 * dist_corr(x_private, z_private).sum()

            total_loss = total_loss + dist_corr_loss
        if self.gan_regularizer and not self.gan_noise:
            self.local_AE_list[client_id].eval()
            output_image = self.local_AE_list[client_id](z_private)
            
            x_private = denormalize(x_private, self.dataset)
            '''MSE is poor in regularization here. unstable. We stick to SSIM'''
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                ssim_term = ssim_loss(output_image, x_private)
                
                if self.ssim_threshold > 0.0:
                    if ssim_term > self.ssim_threshold:
                        gan_loss = self.alpha2 * (ssim_term - self.ssim_threshold) # Let SSIM approaches 0.4 to avoid overfitting
                    else:
                        gan_loss = 0.0 # Let SSIM approaches 0.4 to avoid overfitting
                else:
                    gan_loss = self.alpha2 * ssim_term  
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                mse_term = mse_loss(output_image, x_private)
                gan_loss = - self.alpha2 * mse_term  
            total_loss = total_loss + gan_loss
            
        if self.spurious_activation:  # increase activation dynamic range == increase weight value
            # sp_loss = spurious_loss(self.model.local_list[client_id], self.alpha4)
            sp_loss = spurious_loss(z_private, self.alpha4)
            total_loss = total_loss + sp_loss

        total_loss.backward()

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses

    def validate_target(self, client_id=0):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        if client_id == 0:
            self.f.eval()
        elif client_id == 1:
            self.c.eval()
        elif client_id > 1:
            self.model.local_list[client_id].eval()
        self.f_tail.eval()
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss()

        activation_0 = {}

        def get_activation_0(name):
            def hook(model, input, output):
                activation_0[name] = output.detach()

            return hook
            # with torch.no_grad():

            #     count = 0
            #     for name, m in self.model.cloud.named_modules():
            #         if attack_from_later_layer == count:
            #             m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
            #             valid_key = "ACT-{}".format(name)
            #             break
            #         count += 1
            #     output = self.model.cloud(ir)

            # ir = activation_4[valid_key]

        for name, m in self.model.local_list[client_id].named_modules():
            m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))

        for name, m in self.f_tail.named_modules():
            m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))


        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            activation_0 = {}
            # compute output
            with torch.no_grad():

                output = self.model.local_list[client_id](input)

                # code for save the activation of cutlayer
                

                if self.bhtsne:
                    self.save_activation_bhtsne(output, target, client_id)
                    exit()

                '''Optional, Test validation performance with local_DP/dropout (apply DP during query)'''
                if self.local_DP:
                    if "laplace" in self.regularization_option:
                        noise = torch.from_numpy(
                            np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=output.size())).cuda()
                    else:  # apply gaussian noise
                        delta = 10e-5
                        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                        noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                if self.dropout_defense:
                    output = dropout_defense(output, self.dropout_ratio)
                if self.topkprune:
                    output = prune_defense(output, self.topkprune_ratio)
            if self.gan_noise:
                epsilon = self.alpha2
                
                self.local_AE_list[client_id].eval()
                fake_act = output.clone()
                grad = torch.zeros_like(output).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                input = denormalize(input, self.dataset)

                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, input)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, input)
                    loss.backward()
                    grad += torch.sign(fake_act.grad) 

                output = output - grad.detach() * epsilon
            with torch.no_grad():
                output = self.f_tail(output)

                if "mobilenetv2" in self.arch:
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    output = F.avg_pool2d(output, 8)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                else:
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                loss = criterion(output, target)

            if i == 0:
                try:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)

                    # setup tensorboard
                    if self.save_activation_tensor:
                        save_tensor_path = self.save_dir + "/saved_tensors"
                        if not os.path.isdir(save_tensor_path):
                            os.makedirs(save_tensor_path)
                    for key, value in activation_0.items():
                        if "client" in key:
                            self.writer.add_histogram("local_act/{}".format(key), value.clone().cpu().data.numpy(), i)
                            if self.save_activation_tensor:
                                np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                        if "server" in key:
                            self.writer.add_histogram("server_act/{}".format(key), value.clone().cpu().data.numpy(), i)
                            if self.save_activation_tensor:
                                np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                    
                    for name, m in self.model.local_list[client_id].named_modules():
                        handle = m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))
                        handle.remove()
                    for name, m in self.f_tail.named_modules():
                        handle = m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))
                        handle.remove()
                except:
                    print("something went wrong adding histogram, ignore it..")

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target, compress_V4shadowlabel=self.V4shadowlabel, num_client=self.num_client)[0] #If V4shadowlabel is activated, add one extra step to process output back to orig_class
            prec1 = accuracy(output.data, target)[
                0]  # If V4shadowlabel is activated, add one extra step to process output back to orig_class
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time

            if i % 50 == 0:
                self.logger.debug('Test (client-{0}):\t'
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    client_id, loss=losses,
                    top1=top1))
        for name, param in self.model.local_list[client_id].named_parameters():
            self.writer.add_histogram("local_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        for name, param in self.model.cloud.named_parameters():
            self.writer.add_histogram("server_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'
                          .format(top1=top1))

        return top1.avg, losses.avg

    def infer_path_list(self, path_to_infer):
        split_list = path_to_infer.split("checkpoint_f")
        first_part = split_list[0]
        second_part = split_list[1]
        model_path_list = []
        for i in range(self.num_client):
            if i == 0:
                model_path_list.append(path_to_infer)
            elif i == 1:
                model_path_list.append(first_part + "checkpoint_c" + second_part)
            else:
                model_path_list.append(first_part + "checkpoint_local{}".format(i) + second_part)

        return model_path_list

    def resume(self, model_path_f=None):
        if model_path_f is None:
            try:
                if "V" in self.scheme:
                    checkpoint = torch.load(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                    model_path_list = self.infer_path_list(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                else:
                    checkpoint = torch.load(self.save_dir + "checkpoint_{}.tar".format(self.n_epochs))
                    # model_path_list = self.infer_path_list(self.save_dir + "checkpoint_200.tar")
            except:
                print("No valid Checkpoint Found!")
                return
        else:
            if "V" in self.scheme:
                model_path_list = self.infer_path_list(model_path_f)

        if "V" in self.scheme:
            for i in range(self.num_client):
                print("load client {}'s local".format(i))
                checkpoint_i = torch.load(model_path_list[i])
                self.model.local_list[i].cuda()
                self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
        else:
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.f = self.model.local
            self.f.cuda()

        try:
            self.call_resume = True
            print("load cloud")
            checkpoint = torch.load(self.save_dir + "checkpoint_cloud_{}.tar".format(self.n_epochs))
            self.f_tail.cuda()
            self.f_tail.load_state_dict(checkpoint, strict = False)
            print("load classifier")
            checkpoint = torch.load(self.save_dir + "checkpoint_classifier_{}.tar".format(self.n_epochs))
            self.classifier.cuda()
            self.classifier.load_state_dict(checkpoint, strict = False)
        except:
            print("might be old style saving, load entire model")
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.call_resume = True
            self.f = self.model.local
            self.f.cuda()
            self.f_tail = self.model.cloud
            self.f_tail.cuda()
            self.classifier = self.model.classifier
            self.classifier.cuda()

    def sync_client(self):
        # update global weights
        global_weights = average_weights(self.model.local_list)

        # update global weights
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(global_weights)

    def sync_decoder(self):
        # update global weights
        global_weights = average_weights(self.local_AE_list)

        # update global weights
        for i in range(self.num_client):
            self.local_AE_list[i].load_state_dict(global_weights)

    def gan_train_step(self, input_images, client_id, loss_type="SSIM"):
        device = next(self.model.local_list[client_id].parameters()).device

        input_images = input_images.to(device)

        self.model.local_list[client_id].eval()

        z_private = self.model.local_list[client_id](input_images)

        self.local_AE_list[client_id].train()

        x_private, z_private = Variable(input_images).to(device), Variable(z_private)

        x_private = denormalize(x_private, self.dataset)

        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)

            if loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif loss_type == "MSE":
                MSE_loss = torch.nn.MSELoss()
                loss = MSE_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)
            else:
                raise ("No such loss_type for gan train step")
            
            z_private = z_private - grad.detach() * epsilon
            
            self.local_AE_list[client_id].train()

        output = self.local_AE_list[client_id](z_private.detach())

        if loss_type == "SSIM":
            ssim_loss = pytorch_ssim.SSIM()
            loss = -ssim_loss(output, x_private)
        elif loss_type == "MSE":
            MSE_loss = torch.nn.MSELoss()
            loss = MSE_loss(output, x_private)
        else:
            raise ("No such loss_type for gan train step")
        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].zero_grad()

        loss.backward()

        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].step()

        losses = loss.detach().cpu().numpy()
        del loss

        return losses

    def __call__(self, log_frequency=500, verbose=False, progress_bar=True):
        self.logger.debug("Model's smashed-data size is {}".format(str(self.model.get_smashed_data_size())))
        best_avg_accu = 0.0
        if not self.call_resume:
            LOG = np.zeros((self.n_epochs * self.num_batches, self.num_client))
            client_iterator_list = []
            for client_id in range(self.num_client):
                client_iterator_list.append(iter(self.client_dataloader[client_id]))

            #load pre-train models
            if self.load_from_checkpoint:
                checkpoint_dir = "./pretrained_models/{}_cutlayer_{}_bottleneck_{}_dataset_{}/".format(self.arch, self.cutting_layer, self.bottleneck_option, self.source_task)
                try:
                    checkpoint_i = torch.load(checkpoint_dir + "checkpoint_f_best.tar")
                except:
                    print("No valid Checkpoint Found!")
                    return
                if "V" in self.scheme:
                    for i in range(self.num_client):
                        print("load client {}'s local".format(i))
                        self.model.local_list[i].cuda()
                        self.model.local_list[i].load_state_dict(checkpoint_i)
                else:
                    self.model.cuda()
                    self.model.local.load_state_dict(checkpoint_i)
                    self.f = self.model.local
                    self.f.cuda()
                
                load_classfier = False
                if self.load_from_checkpoint_server:
                    print("load cloud")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_cloud_best.tar")
                    self.f_tail.cuda()
                    self.f_tail.load_state_dict(checkpoint)
                if load_classfier:
                    print("load classifier")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_classifier_best.tar")
                    self.classifier.cuda()
                    self.classifier.load_state_dict(checkpoint)

            if self.gan_regularizer:
                self.pre_GAN_train(30, range(self.num_client))


            self.logger.debug("Real Train Phase: done by all clients, for total {} epochs".format(self.n_epochs))

            if self.save_more_checkpoints:
                epoch_save_list = [1, 2 ,5 ,10 ,20 ,50 ,100]
            else:
                epoch_save_list = []
            # If optimize_computation, set GAN updating frequency to 1/5.
            ssim_log = 0.
            
            interval = self.optimize_computation
            self.logger.debug("GAN training interval N (once every N step) is set to {}!".format(interval))
            
            
            #Main Training
            for epoch in range(1, self.n_epochs+1):
                if epoch > self.warm:
                    self.scheduler_step(epoch)
                    if self.gan_regularizer:
                        self.gan_scheduler_step(epoch)
                
                if self.client_sample_ratio  == 1.0:
                    idxs_users = range(self.num_client)
                else:
                    idxs_users = np.random.choice(range(self.actual_num_users), self.num_client, replace=False) # 10 out of 1000
                
                client_iterator_list = []
                for client_id in range(self.num_client):
                    client_iterator_list.append(iter(self.client_dataloader[idxs_users[client_id]]))

                self.logger.debug("Train in {} style".format(self.scheme))
                if "epoch" in self.scheme:
                    for batch in range(self.num_batches):

                        # shuffle_client_list = range(self.num_client)
                        for client_id in range(self.num_client):
                            if self.scheme == "V1_epoch" or self.scheme == "V3_epoch":
                                self.optimizer_zero_grad()
                            # Get data

                            try:
                                images, labels = next(client_iterator_list[client_id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    images, labels = next(client_iterator_list[client_id])
                            except StopIteration:
                                client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                images, labels = next(client_iterator_list[client_id])

                            # Train the AE decoder if self.gan_regularizer is enabled:
                            if self.gan_regularizer and batch % interval == 0:
                                for i in range(self.gan_num_step):
                                    ssim_log = -self.gan_train_step(images, client_id, loss_type=self.gan_loss_type)  # orig_epoch_gan_train

                            if self.scheme == "V2_epoch" or self.scheme == "V4_epoch" or self.scheme == "orig_epoch":
                                self.optimizer_zero_grad()
                            
                            # Train step (client/server)
                            train_loss, f_loss = self.train_target_step(images, labels, client_id)
                            
                            if self.scheme == "V2_epoch" or self.scheme == "V4_epoch" or self.scheme == "orig_epoch":
                                self.optimizer_step()
                            
                            # Logging
                            # LOG[batch, client_id] = train_loss
                            if verbose and batch % log_frequency == 0:
                                self.logger.debug(
                                    "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                        epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss, f_loss))
                                if self.gan_regularizer:
                                    self.logger.debug(
                                        "log--[{}/{}][{}/{}][client-{}] Adversarial Loss of local AE: {:1.4f}".format(epoch,
                                                                                                                self.n_epochs,
                                                                                                                batch,
                                                                                                                self.num_batches,
                                                                                                                client_id,
                                                                                                                ssim_log))
                            if batch == 0:
                                self.writer.add_scalar('train_loss/client-{}/total'.format(client_id), train_loss,
                                                        epoch)
                                self.writer.add_scalar('train_loss/client-{}/cross_entropy'.format(client_id), f_loss,
                                                        epoch)
                        # Update parameter at the end of a global batch (aggregating all gradients incurred by all clients on server-side model)
                        if self.scheme == "V1_batch" or self.scheme == "V3_batch":
                            self.optimizer_step()
                        
                else:
                    # orig/V1/V2/V3 training, batch-wise sync
                    for batch in range(self.num_batches):

                        if self.scheme == "V1_batch" or self.scheme == "V3_batch":
                            self.optimizer_zero_grad()
                        for client_id in range(self.num_client):
                            # Get data
                            try:
                                images, labels = next(client_iterator_list[client_id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    images, labels = next(client_iterator_list[client_id])
                            except StopIteration:
                                client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                images, labels = next(client_iterator_list[client_id])

                            if self.scheme == "V2_batch" or self.scheme == "V4_batch" or self.scheme == "orig_batch":  # In V2, the server-side model will update sequentially
                                self.optimizer_zero_grad()

                            # Train step (client/server)
                            train_loss, f_loss = self.train_target_step(images, labels, client_id)

                            # Train the AE decoder if self.gan_regularizer is enabled:
                            if self.gan_regularizer and batch % interval == 0:
                                for i in range(self.gan_num_step):
                                    ssim_log = -self.gan_train_step(images, client_id, loss_type=self.gan_loss_type)  # other_scheme_gan_train
                            
                            # If V2/V4/orig_batch, update client/server paramter immediately after completing the forward
                            if self.scheme == "V2_batch" or self.scheme == "V4_batch" or self.scheme == "orig_batch":  # In V2, the server-side model will update sequentially
                                self.optimizer_step(set_client=True, client_id=client_id)

                            # Logging
                            if verbose and batch % log_frequency == 0:
                                self.logger.debug(
                                    "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                        epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss, f_loss))

                                if self.gan_regularizer:
                                    self.logger.debug(
                                        "log--[{}/{}][{}/{}][client-{}] Adversarial Loss of local AE: {:1.4f}".format(epoch,
                                                                                                                self.n_epochs,
                                                                                                                batch,
                                                                                                                self.num_batches,
                                                                                                                client_id,
                                                                                                                ssim_log))
                            if batch == 0:
                                self.writer.add_scalar('train_loss/client-{}/total'.format(client_id), train_loss,
                                                       epoch)
                                self.writer.add_scalar('train_loss/client-{}/cross_entropy'.format(client_id), f_loss,
                                                       epoch)

                        # Update parameter at the end of a global batch (aggregating all gradients incurred by all clients on server-side model)
                        if self.scheme == "V1_batch" or self.scheme == "V3_batch":
                            self.optimizer_step()
                        # V1/V2 synchronization
                        if self.scheme == "V1_batch" or self.scheme == "V2_batch":
                            self.sync_client()
                            if self.gan_regularizer and self.decoder_sync:
                                self.sync_decoder()

                # V1/V2 synchronization
                if self.scheme == "V1_epoch" or self.scheme == "V2_epoch":
                    self.sync_client()
                    if self.gan_regularizer and self.decoder_sync:
                        self.sync_decoder()

                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)


                # Validate and get average accu among clients
                avg_accu = 0
                for client_id in range(self.num_client):
                    accu, loss = self.validate_target(client_id=client_id)
                    self.writer.add_scalar('valid_loss/client-{}/cross_entropy'.format(client_id), loss, epoch)
                    avg_accu += accu
                avg_accu = avg_accu / self.num_client

                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    best_avg_accu = avg_accu

                # Save Model regularly
                if epoch % 50 == 0 or epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)

        if not self.call_resume:
            self.logger.debug("Best Average Validation Accuracy is {}".format(best_avg_accu))
        else:
            LOG = None
            avg_accu = 0
            for client_id in range(self.num_client):
                accu, loss = self.validate_target(client_id=client_id)
                avg_accu += accu
            avg_accu = avg_accu / self.num_client
            self.logger.debug("Best Average Validation Accuracy is {}".format(avg_accu))
        return LOG

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"

        if "V" in self.scheme:
            torch.save(self.f.state_dict(), self.save_dir + 'checkpoint_f_{}.tar'.format(epoch))
            if self.num_client > 1:
                torch.save(self.c.state_dict(), self.save_dir + 'checkpoint_c_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))
            if self.num_client > 2:
                for i in range(2, self.num_client):
                    torch.save(self.model.local_list[i].state_dict(),
                                self.save_dir + 'checkpoint_local{}_{}.tar'.format(i, epoch))
        else:
            torch.save(self.model.state_dict(), self.save_dir + 'checkpoint_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))

    def gen_ir(self, val_single_loader, local_model, img_folder="./tmp", intermed_reps_folder="./tmp", all_label=True,
               select_label=0, attack_from_later_layer=-1, attack_option = "MIA"):
        """
        Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
        """

        # switch to evaluate mode
        local_model.eval()
        file_id = 0
        for i, (input, target) in enumerate(val_single_loader):
            # input = input.cuda(async=True)
            input = input.cuda()
            target = target.item()
            if not all_label:
                if target != select_label:
                    continue

            img_folder = os.path.abspath(img_folder)
            intermed_reps_folder = os.path.abspath(intermed_reps_folder)
            if not os.path.isdir(intermed_reps_folder):
                os.makedirs(intermed_reps_folder)
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            # compute output
            with torch.no_grad():
                ir = local_model(input)
            
            if self.confidence_score:
                self.model.cloud.eval()
                ir = self.model.cloud(ir)
                if "mobilenetv2" in self.arch:
                    ir = F.avg_pool2d(ir, 4)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    ir = F.avg_pool2d(ir, 8)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                else:
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
            
            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_4 = {}

                def get_activation_4(name):
                    def hook(model, input, output):
                        activation_4[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_4 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(ir)
                try:
                    ir = activation_4[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            ir = ir.float()

            if "truncate" in attack_option:
                try:
                    percentage_left = int(attack_option.split("truncate")[1])
                except:
                    print("auto extract percentage fail. Use default percentage_left = 20")
                    percentage_left = 20
                ir = prune_top_n_percent_left(ir, percentage_left)

            inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
            out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
            
            input = denormalize(input, self.dataset)
            save_image(input, inp_img_path)
            torch.save(ir.cpu(), out_tensor_path)
            file_id += 1
        print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                   int(file_id * 0.1)))

    def pre_GAN_train(self, num_epochs, select_client_list=[0]):

        # Generate latest images/activation pair for all clients:
        client_iterator_list = []
        for client_id in range(self.num_client):
            client_iterator_list.append(iter(self.client_dataloader[client_id]))
        try:
            images, labels = next(client_iterator_list[client_id])
            if images.size(0) != self.batch_size:
                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                images, labels = next(client_iterator_list[client_id])
        except StopIteration:
            client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
            images, labels = next(client_iterator_list[client_id])

        for client_id in select_client_list:
            self.save_image_act_pair(images, labels, client_id, 0, clean_option=True)

        for client_id in select_client_list:

            attack_batchsize = 32
            attack_num_epochs = num_epochs
            model_log_file = self.save_dir + '/MIA_attack_{}_{}.log'.format(client_id, client_id)
            logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), client_id, client_id),
                                  model_log_file, level=logging.DEBUG)
            # pass
            image_data_dir = self.save_dir + "/img"
            tensor_data_dir = self.save_dir + "/img"

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)

            if self.dataset == "cifar100":
                val_single_loader, _, _ = get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "cifar10":
                val_single_loader, _, _ = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "svhn":
                val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "mnist":
                _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "fmnist":
                _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "facescrub":
                _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
                    
            attack_path = self.save_dir + '/MIA_attack_{}to{}'.format(client_id, client_id)
            if not os.path.isdir(attack_path):
                os.makedirs(attack_path)
                os.makedirs(attack_path + "/train")
                os.makedirs(attack_path + "/test")
                os.makedirs(attack_path + "/tensorboard")
                os.makedirs(attack_path + "/sourcecode")
            train_output_path = "{}/train".format(attack_path)
            test_output_path = "{}/test".format(attack_path)
            tensorboard_path = "{}/tensorboard/".format(attack_path)
            model_path = "{}/model.pt".format(attack_path)
            path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                         "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

            logger.debug("Generating IR ...... (may take a while)")

            self.gen_ir(val_single_loader, self.model.local_list[client_id], image_data_dir, tensor_data_dir)

            decoder = self.local_AE_list[client_id]

            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            # Construct a dataset for training the decoder
            trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

            # Do real test on target's client activation (and test with target's client ground-truth.)
            sp_testloader = apply_transform_test(1,
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0),
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0))

            # Perform Input Extraction Attack
            self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                        attack_batchsize)
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class)

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)

    def MIA_attack(self, num_epochs, attack_option="MIA", collude_client=1, target_client=0, noise_aware=False,
                   loss_type="MSE", attack_from_later_layer=-1, MIA_optimizer = "Adam", MIA_lr = 1e-3):
        attack_option = attack_option
        MIA_optimizer = MIA_optimizer
        MIA_lr = MIA_lr
        attack_batchsize = 32
        attack_num_epochs = num_epochs
        model_log_file = self.save_dir + '/{}_attack_{}_{}.log'.format(attack_option, collude_client, target_client)
        logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), collude_client, target_client),
                              model_log_file, level=logging.DEBUG)
        # pass
        image_data_dir = self.save_dir + "/img"
        tensor_data_dir = self.save_dir + "/img"

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)

        if self.dataset == "cifar100":
            val_single_loader, _, _ = get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "cifar10":
            val_single_loader, _, _ = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "svhn":
            val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "mnist":
            _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "fmnist":
            _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "facescrub":
            _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)

        attack_path = self.save_dir + '/{}_attack_{}to{}'.format(attack_option, collude_client, target_client)
        if not os.path.isdir(attack_path):
            os.makedirs(attack_path)
            os.makedirs(attack_path + "/train")
            os.makedirs(attack_path + "/test")
            os.makedirs(attack_path + "/tensorboard")
            os.makedirs(attack_path + "/sourcecode")
        train_output_path = "{}/train".format(attack_path)
        test_output_path = "{}/test".format(attack_path)
        tensorboard_path = "{}/tensorboard/".format(attack_path)
        model_path = "{}/model.pt".format(attack_path)
        path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                     "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

        if ("MIA" in attack_option) and ("MIA_mf" not in attack_option):
            logger.debug("Generating IR ...... (may take a while)")

            self.gen_ir(val_single_loader, self.model.local_list[collude_client], image_data_dir, tensor_data_dir,
                    attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
            
            for filename in os.listdir(tensor_data_dir):
                if ".pt" in filename:
                    sampled_tensor = torch.load(tensor_data_dir + "/" + filename)
                    input_nc = sampled_tensor.size()[1]
                    try:
                        input_dim = sampled_tensor.size()[2]
                    except:
                        print("Extract input dimension fialed, set to 0")
                        input_dim = 0
                    break

            if self.gan_AE_type == "custom":
                decoder = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                  activation=self.gan_AE_activation).cuda()
            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()

            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            
            else:
                raise ("No such GAN AE type.")

            if self.measure_option:
                noise_input = torch.randn([1, input_nc, input_dim, input_dim])
                device = next(decoder.parameters()).device
                noise_input = noise_input.to(device)
                macs, num_param = profile(decoder, inputs=(noise_input,))
                self.logger.debug(
                    "{} Decoder Model's Mac and Param are {} and {}".format(self.gan_AE_type, macs, num_param))
                
                '''Uncomment below to also get decoder's inference and training time overhead.'''
                # decoder.cpu()
                # noise_input = torch.randn([128, input_nc, input_dim, input_dim])
                # with torch.no_grad():
                #     _ = decoder(noise_input)
                #     start_time = time.time()
                #     for _ in range(500):  # CPU warm up
                #         _ = decoder(noise_input)
                #     lapse_cpu_decoder = (time.time() - start_time) / 500
                # self.logger.debug("Decoder Model's Inference time on CPU is {}".format(lapse_cpu_decoder))

                # criterion = torch.nn.MSELoss()
                # noise_reconstruction = torch.randn([128, 3, 32, 32])
                # reconstruction = decoder(noise_input)

                # r_loss = criterion(reconstruction, noise_reconstruction)
                # r_loss.backward()
                # lapse_cpu_decoder_train = 0
                # for _ in range(500):  # CPU warm up
                #     reconstruction = decoder(noise_input)
                #     r_loss = criterion(reconstruction, noise_reconstruction)
                #     start_time = time.time()
                #     r_loss.backward()
                #     lapse_cpu_decoder_train += (time.time() - start_time)
                # lapse_cpu_decoder_train = lapse_cpu_decoder_train / 500
                # del r_loss, reconstruction, noise_input
                # self.logger.debug("Decoder Model's Train time on CPU is {}".format(lapse_cpu_decoder_train))
                # decoder.cuda()

            '''Setting attacker's learning algorithm'''
            # optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            if MIA_optimizer == "Adam":
                optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr)
            elif MIA_optimizer == "SGD":
                optimizer = torch.optim.SGD(decoder.parameters(), lr=MIA_lr)
            else:
                raise("MIA optimizer {} is not supported!".format(MIA_optimizer))
            # Construct a dataset for training the decoder
            trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

            # Do real test on target's client activation (and test with target's client ground-truth.)
            sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                          self.n_epochs))
            if "gan_adv_noise" in self.regularization_option and noise_aware:
                print("create a second decoder")
                if self.gan_AE_type == "custom":
                    decoder2 = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                       output_dim=32, activation=self.gan_AE_activation).cuda()
                elif "conv_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("conv_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from conv_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()
                elif "res_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("res_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from res_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()
                else:
                    raise ("No such GAN AE type.")
                # optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                self.attack(attack_num_epochs, decoder2, optimizer2, trainloader, testloader, logger, path_dict,
                            attack_batchsize, pretrained_decoder=self.local_AE_list[collude_client], noise_aware=noise_aware)
                decoder = decoder2  # use decoder2 for testing
            else:
                # Perform Input Extraction Attack
                self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                            attack_batchsize, noise_aware=noise_aware, loss_type=loss_type)

            
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class)

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)
            return mse_score, ssim_score, psnr_score
        elif attack_option == "MIA_mf":  # Stands for Model-free MIA, does not need a AE model, optimize each fake image instead.

            lambda_TV = 0.0
            lambda_l2 = 0.0
            num_step = attack_num_epochs * 60

            sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                          self.n_epochs))
            criterion = nn.MSELoss().cuda()
            ssim_loss = pytorch_ssim.SSIM()
            all_test_losses = AverageMeter()
            ssim_test_losses = AverageMeter()
            psnr_test_losses = AverageMeter()
            fresh_option = True
            for num, data in enumerate(sp_testloader, 1):
                img, ir, _ = data

                # optimize a fake_image to (1) have similar ir, (2) have small total variance, (3) have small l2
                img = img.cuda()
                if not fresh_option:
                    ir = ir.cuda()
                self.model.local_list[collude_client].eval()
                self.model.local_list[target_client].eval()

                fake_image = torch.zeros(img.size(), requires_grad=True, device="cuda")
                optimizer = torch.optim.Adam(params=[fake_image], lr=8e-1, amsgrad=True, eps=1e-3)
                # optimizer = torch.optim.Adam(params = [fake_image], lr = 1e-2, amsgrad=True, eps=1e-3)
                for step in range(1, num_step + 1):
                    optimizer.zero_grad()

                    fake_ir = self.model.local_list[collude_client](fake_image)  # Simulate Original

                    if fresh_option:
                        ir = self.model.local_list[target_client](img)  # Getting fresh ir from target local model

                    featureLoss = criterion(fake_ir, ir)

                    TVLoss = TV(fake_image)
                    normLoss = l2loss(fake_image)

                    totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

                    totalLoss.backward()

                    optimizer.step()
                    # if step % 100 == 0:
                    if step == 0 or step == num_step:
                        logger.debug("Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(step,
                                                                                             featureLoss.cpu().detach().numpy(),
                                                                                             TVLoss.cpu().detach().numpy(),
                                                                                             normLoss.cpu().detach().numpy()))
                imgGen = fake_image.clone()
                imgOrig = img.clone()

                mse_loss = criterion(imgGen, imgOrig)
                ssim_loss_val = ssim_loss(imgGen, imgOrig)
                psnr_loss_val = get_PSNR(imgOrig, imgGen)
                all_test_losses.update(mse_loss.item(), ir.size(0))
                ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
                psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
                if not os.path.isdir(test_output_path + "/{}".format(attack_num_epochs)):
                    os.mkdir(test_output_path + "/{}".format(attack_num_epochs))
                
                torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(attack_num_epochs,
                                                                                                 num * attack_batchsize + attack_batchsize))
                
                torchvision.utils.save_image(imgOrig, test_output_path + '/{}/inp_{}.jpg'.format(attack_num_epochs,
                                                                                                 num * attack_batchsize + attack_batchsize))
            logger.debug("MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                all_test_losses.avg))
            logger.debug("SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                ssim_test_losses.avg))
            logger.debug("PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                psnr_test_losses.avg))
            return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    def attack(self, num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict, batch_size,
               loss_type="MSE", pretrained_decoder=None, noise_aware=False):
        round_ = 0
        min_val_loss = 999.
        max_val_loss = 0.
        train_output_freq = 10
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # Optimize based on MSE distance
        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "SSIM":
            criterion = pytorch_ssim.SSIM()
        elif loss_type == "PSNR":
            criterion = None
        else:
            raise ("No such loss in self.attack")
        device = next(decoder.parameters()).device
        decoder.train()
        for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
            for num, data in enumerate(trainloader, 1):
                img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)
                # print(img)
                # Use local DP for training the AE.
                if self.local_DP and noise_aware:
                    with torch.no_grad():
                        if "laplace" in self.regularization_option:
                            ir += torch.from_numpy(
                                np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=ir.size())).cuda()
                        else:  # apply gaussian noise
                            delta = 10e-5
                            sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                            ir += sigma * torch.randn_like(ir).cuda()
                if self.dropout_defense and noise_aware:
                    ir = dropout_defense(ir, self.dropout_ratio)
                if self.topkprune and noise_aware:
                    ir = prune_defense(ir, self.topkprune_ratio)
                if pretrained_decoder is not None and "gan_adv_noise" in self.regularization_option and noise_aware:
                    epsilon = self.alpha2
                    
                    pretrained_decoder.eval()
                    fake_act = ir.clone()
                    grad = torch.zeros_like(ir).cuda()
                    fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                    x_recon = pretrained_decoder(fake_act)
                    if self.gan_loss_type == "SSIM":
                        ssim_loss = pytorch_ssim.SSIM()
                        loss = ssim_loss(x_recon, img)
                        loss.backward()
                        grad -= torch.sign(fake_act.grad)
                    else:
                        mse_loss = nn.MSELoss()
                        loss = mse_loss(x_recon, img)
                        loss.backward()
                        grad += torch.sign(fake_act.grad)
                    ir = ir + grad.detach() * epsilon
                # print(ir.size())
                output = decoder(ir)

                if loss_type == "MSE":
                    reconstruction_loss = criterion(output, img)
                elif loss_type == "SSIM":
                    reconstruction_loss = -criterion(output, img)
                elif loss_type == "PSNR":
                    reconstruction_loss = -1 / 10 * get_PSNR(img, output)
                else:
                    raise ("No such loss in self.attack")
                train_loss = reconstruction_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses.update(train_loss.item(), ir.size(0))

            if (epoch + 1) % train_output_freq == 0:
                save_images(img, output, epoch, path_dict["train_output_path"], offset=0, batch_size=batch_size)

            for num, data in enumerate(testloader, 1):
                img, ir = data

                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)

                output = decoder(ir)

                reconstruction_loss = criterion(output, img)
                val_loss = reconstruction_loss

                if loss_type == "MSE" and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "SSIM" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "PSNR" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                val_losses.update(val_loss.item(), ir.size(0))

                self.writer.add_scalar('decoder_loss/val', val_loss.item(), len(testloader) * epoch + num)
                self.writer.add_scalar('decoder_loss/val_loss/reconstruction', reconstruction_loss.item(),
                                       len(testloader) * epoch + num)

            for name, param in decoder.named_parameters():
                self.writer.add_histogram("decoder_params/{}".format(name), param.clone().cpu().data.numpy(), epoch)

            # torch.save(decoder.state_dict(), path_dict["model_path"])
            logger.debug(
                "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                    epoch + 1,
                    num_epochs, train_losses=train_losses, val_losses=val_losses))
        if loss_type == "MSE":
            logger.debug("Best Validation Loss is {}".format(min_val_loss))
        elif loss_type == "SSIM":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))
        elif loss_type == "PSNR":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))

    def test_attack(self, num_epochs, decoder, sp_testloader, logger, path_dict, batch_size, num_classes=10,
                    select_label=0):
        device = next(decoder.parameters()).device
        # print("Load the best Decoder Model...")
        new_state_dict = torch.load(path_dict["model_path"])
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        # test_losses = []
        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()
        ssim_loss = pytorch_ssim.SSIM()

        criterion = nn.MSELoss()

        for num, data in enumerate(sp_testloader, 1):
            img, ir, label = data

            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)
            output_imgs = decoder(ir)
            reconstruction_loss = criterion(output_imgs, img)
            ssim_loss_val = ssim_loss(output_imgs, img)
            psnr_loss_val = get_PSNR(img, output_imgs)
            all_test_losses.update(reconstruction_loss.item(), ir.size(0))
            ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
            psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
            save_images(img, output_imgs, num_epochs, path_dict["test_output_path"], offset=num, batch_size=batch_size)

        logger.debug(
            "MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(all_test_losses.avg))
        logger.debug(
            "SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(ssim_test_losses.avg))
        logger.debug(
            "PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(psnr_test_losses.avg))
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    def save_activation_bhtsne(self, save_activation, target, client_id):
        """
            Run one train epoch
        """

        path_dir = os.path.join(self.save_dir, 'save_activation_cutlayer')
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        save_activation = save_activation.float()
        save_activation = save_activation.cpu().numpy()
        save_activation = save_activation.reshape(self.batch_size, -1)
        np.savetxt(os.path.join(path_dir, "{}.txt".format(client_id)), save_activation, fmt='%.2f')

        target = target.float()
        target = target.cpu().numpy()
        target = target.reshape(self.batch_size, -1)
        np.savetxt(os.path.join(path_dir, "{}target.txt".format(client_id)), target, fmt='%.2f')

    #Generate test set for MIA decoder
    def save_image_act_pair(self, input, target, client_id, epoch, clean_option=False, attack_from_later_layer=-1, attack_option = "MIA"):
        """
            Run one train epoch
        """
        path_dir = os.path.join(self.save_dir, 'save_activation_client_{}_epoch_{}'.format(client_id, epoch))
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        else:
            rmtree(path_dir)
            os.makedirs(path_dir)
        input = input.cuda()

        for j in range(input.size(0)):
            img = input[None, j, :, :, :]
            label = target[None, j]
            with torch.no_grad():
                if client_id == 0:
                    self.f.eval()
                    save_activation = self.f(img)
                elif client_id == 1:
                    self.c.eval()
                    save_activation = self.c(img)
                elif client_id > 1:
                    self.model.local_list[client_id].eval()
                    save_activation = self.model.local_list[client_id](img)
                if self.confidence_score:
                    self.model.cloud.eval()
                    save_activation = self.model.cloud(save_activation)
                    if "mobilenetv2" in self.arch:
                        save_activation = F.avg_pool2d(save_activation, 4)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    elif self.arch == "resnet20" or self.arch == "resnet32":
                        save_activation = F.avg_pool2d(save_activation, 8)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    else:
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
            

            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_3 = {}

                def get_activation_3(name):
                    def hook(model, input, output):
                        activation_3[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_3 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_3("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(save_activation)
                try:
                    save_activation = activation_3[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            if self.local_DP and not clean_option:  # local DP or additive noise
                if "laplace" in self.regularization_option:
                    save_activation += torch.from_numpy(
                        np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=save_activation.size())).cuda()
                    # the addtive work uses scale in (0.1 0.5 1.0) -> (1 2 10) regularization_strength (self.dp_epsilon)
                else:  # apply gaussian noise
                    delta = 10e-5
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                    save_activation += sigma * torch.randn_like(save_activation).cuda()
            if self.dropout_defense and not clean_option:  # activation dropout defense
                save_activation = dropout_defense(save_activation, self.dropout_ratio)
            if self.topkprune and not clean_option:
                save_activation = prune_defense(save_activation, self.topkprune_ratio)
            
            img = denormalize(img, self.dataset)
                
            if self.gan_noise and not clean_option:
                epsilon = self.alpha2
                self.local_AE_list[client_id].eval()
                fake_act = save_activation.clone()
                grad = torch.zeros_like(save_activation).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, img)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, img)
                    loss.backward()
                    grad += torch.sign(fake_act.grad)  

                save_activation = save_activation - grad.detach() * epsilon
            if "truncate" in attack_option:
                save_activation = prune_top_n_percent_left(save_activation)
            
            save_activation = save_activation.float()
            
            save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
            torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
            torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))


# if __name__ == "__main__":
#     print(test_denorm())