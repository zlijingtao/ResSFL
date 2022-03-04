# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import MIA_torch
from datasets_torch import *
from utils import setup_logger
import argparse

import torch



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# match below exactlly the same with the training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=4, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', required=True, type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=1, type=int, help='number of client')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--test_best', action='store_true', default=False, help='if True, test the best epoch')
parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--random_seed', default=123, type=int, help='random_seed for the testing dataset')
parser.add_argument('--scheme', default="V2_epoch", type=str, help='the name of the scheme, either V3 or others')
parser.add_argument('--bottleneck_option', default="None", type=str, help='set bottleneck option')

# test setting
parser.add_argument('--regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0.0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--average_time', default=1, type=int, help='number of time to run the MIA attack for an average performance')
parser.add_argument('--target_client', default=0, type=int, help='id of the target client')
parser.add_argument('--attack_scheme', default="MIA", type=str, help='the name of the attack scheme, either MIA or MIA_mf')
parser.add_argument('--attack_epochs', default=50, type=int, help='number of epochs for the MIA attack algorithm')
parser.add_argument('--attack_from_later_layer', default=-1, type=int, help='set to greater than -1 if attacking at a later layer')
parser.add_argument('--gan_AE_type', default="custom", type=str, help='the name of the AE used in attack, option: custom, simple, simplest')
parser.add_argument('--attack_loss_type', default="MSE", type=str, help='the type of the loss function used in attack, option: MSE, SSIM')
parser.add_argument('--gan_loss_type', default="SSIM", type=str, help='loss type of training defensive decoder: SSIM or MSE')
parser.add_argument('--MIA_optimizer', default="Adam", type=str, help='the type of the learning algorithm used in attack, option: Adam, SGD')
parser.add_argument('--MIA_lr', default=1e-3, type=float, help='learning rate used in attack.')
parser.add_argument('--save_activation_tensor', action='store_true', default=False, help='if True, we save_activation_tensor.')
parser.add_argument('--attack_confidence_score', action='store_true', default=False, help='if True, we attack confidence score.')
parser.add_argument('--measure_option', action='store_true', default=False, help='if True, we measure the inference and training latency')
parser.add_argument('--noise_aware', action='store_true', default=False, help='if True, we set noise_aware and try to break GAN_noise defense.')
parser.add_argument('--new_log_folder', action='store_true', default=False, help='if True, we set separate folder to store log results name: $regularization_$regularization_strength.')
parser.add_argument('--bhtsne_option', action='store_true', default=False, help='if True, we set bhtsne_option to visualize activation.')

args = parser.parse_args()

batch_size = args.batch_size
cutting_layer = args.cutlayer
date_list = []
date_list.append(args.filename)
num_client = args.num_client
target_client = args.target_client
mse_score_list = []
ssim_score_list = []
psnr_score_list = []
random_seed_list = range(123, 123 + args.average_time)


torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if args.attack_confidence_score:
    args.attack_from_later_layer = -1


for date_0 in date_list:

    if args.test_best:
        args.num_epochs = "best"
    save_dir_name = "./{}/{}".format(args.folder, date_0)
    mi = MIA_torch.MIA(args.arch, cutting_layer, batch_size, n_epochs = args.num_epochs, scheme = args.scheme, 
                        num_client = num_client, dataset=args.dataset, save_dir= save_dir_name, 
                        gan_AE_type = args.gan_AE_type, regularization_option=args.regularization, regularization_strength = args.regularization_strength, 
                        random_seed = args.random_seed, bottleneck_option = args.bottleneck_option,
                        measure_option = args.measure_option, bhtsne_option = args.bhtsne_option, attack_confidence_score = args.attack_confidence_score, 
                        save_activation_tensor = args.save_activation_tensor, gan_loss_type = args.gan_loss_type)

    if args.new_log_folder:
        new_folder_dir = mi.save_dir + '/{}_{}/'.format(args.regularization, args.regularization_strength)
        new_folder_dir = os.path.abspath(new_folder_dir)
        if not os.path.isdir(new_folder_dir):
            os.makedirs(new_folder_dir)
        model_log_file = new_folder_dir + '/MIA.log'
        mi.logger = setup_logger('{}_logger'.format(str(save_dir_name)), model_log_file, level=logging.DEBUG)

    if args.measure_option:
        # Print out the number of MAC operations and Params
        c_mac, c_num_param, s_mac, s_num_param = mi.model.get_MAC_param()
        mi.logger.debug("Client Model's Mac and Param are {} and {}".format(c_mac, c_num_param))
        mi.logger.debug("Server Model's Mac and Param are {} and {}".format(s_mac, s_num_param))
        # cpu_time, gpu_time = mi.model.get_inference_time()
        # mi.logger.debug("Client-side model Inference on cpu is {}, Server-side model Inference on gpu is {}".format(cpu_time, gpu_time))

        # cpu_time, _ = mi.model.get_inference_time(federated = True)
        # mi.logger.debug("Entire model Inference on cpu is {}, mimicing federated learning case".format(cpu_time))


        # cpu_time, gpu_time = mi.model.get_train_time()
        # mi.logger.debug("Client-side model Train on cpu is {}, Server-side model Train on gpu is {}".format(cpu_time, gpu_time))

        # cpu_time, _ = mi.model.get_train_time(federated = True)
        # mi.logger.debug("Entire model Training on cpu is {}, mimicing federated learning case".format(cpu_time))
    
    if "orig" not in args.scheme:
        mi.resume("./{}/{}/checkpoint_f_{}.tar".format(args.folder, date_0, args.num_epochs))
    else:
        print("resume orig scheme's checkpoint")
        mi.resume("./{}/{}/checkpoint_{}.tar".format(args.folder, date_0, args.num_epochs))

    if "gan_adv_noise" in args.regularization:
        mi.logger.debug("regularization_strength for GAN_noise is {}".format(args.regularization_strength))
        mi.pre_GAN_train(30)
        noise_aware = args.noise_aware #set to False to bypass the noise-aware training, set to True is default.
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break GAN_noise Defense) ==")
    elif "local_dp" in args.regularization:
        noise_aware = args.noise_aware #set to False to bypass the noise-aware training, set to True is default.
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    elif "dropout" in args.regularization:
        noise_aware = args.noise_aware #set to False to bypass the noise-aware training, set to True is default.
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    elif "topkprune" in args.regularization:
        noise_aware = args.noise_aware #set to False to bypass the noise-aware training, set to True is default.
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    else:
        noise_aware = False
    # '''Generate random images/activation pair:'''
    # if mi.num_client > 1:
    #     client_iterator_list = []
    #     for client_id in range(mi.num_client):
    #         client_iterator_list.append(iter(mi.client_dataloader[client_id]))
    # else:
    #     client_iterator_list = [iter(mi.client_dataloader)]
    # client_id = 0
    # print(next(client_iterator_list[client_id]))
    # try:
    #     images, labels = next(client_iterator_list[client_id])
    #     if images.size(0) != mi.batch_size:
    #         client_iterator_list[client_id] = iter(mi.client_dataloader[client_id])
    #         images, labels = next(client_iterator_list[client_id])
    # except StopIteration:
    #     client_iterator_list[client_id] = iter(mi.client_dataloader[client_id])
    #     images, labels = next(client_iterator_list[client_id])

    # torch.save(images, "./test_fmnist_image.pt")
    # torch.save(labels, "./test_fmnist_label.pt")

    '''Use a fix set of testing image for each experiment'''
    if args.dataset == "cifar10":
        images = torch.load("./test_cifar10_image.pt")
        labels = torch.load("./test_cifar10_label.pt")
    elif args.dataset == "svhn":
        images = torch.load("./test_svhn_image.pt")
        labels = torch.load("./test_svhn_label.pt")
    elif args.dataset == "cifar100":
        images = torch.load("./test_cifar100_image.pt")
        labels = torch.load("./test_cifar100_label.pt")
    elif args.dataset == "mnist":
        images = torch.load("./test_mnist_image.pt")
        labels = torch.load("./test_mnist_label.pt")
    elif args.dataset == "fmnist":
        images = torch.load("./test_fmnist_image.pt")
        labels = torch.load("./test_fmnist_label.pt")
    elif args.dataset == "facescrub":
        images = torch.load("./test_facescrub_image.pt")
        labels = torch.load("./test_facescrub_label.pt")

    for client_id in range(mi.num_client):
        mi.save_image_act_pair(images, labels, client_id, args.num_epochs, attack_from_later_layer= args.attack_from_later_layer, attack_option= args.attack_scheme)


    log_frequency = 500
    skip_valid = False
    if not skip_valid:
        LOG = mi(verbose=True, progress_bar=True, log_frequency=log_frequency)

    # mi.train_GAN()


    for random_seed in random_seed_list:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        client_mse_list = []
        client_ssim_list = []
        client_psnr_list = []
        for j in range(num_client):
            if num_client > 1 and j == target_client: #if j == target_client:
                continue
            mse_score, ssim_score, psnr_score = mi.MIA_attack(args.attack_epochs, attack_option=args.attack_scheme, collude_client=j, target_client=target_client, noise_aware = noise_aware, loss_type = args.attack_loss_type, attack_from_later_layer = args.attack_from_later_layer, MIA_optimizer=args.MIA_optimizer, MIA_lr=args.MIA_lr)
            client_mse_list.append(mse_score)
            client_ssim_list.append(ssim_score)
            client_psnr_list.append(psnr_score)
        
        mse_score_list.append(np.min(np.array(client_mse_list)))
        ssim_score_list.append(np.max(np.array(client_ssim_list)))
        psnr_score_list.append(np.max(np.array(client_psnr_list)))
    
    avg_mse_score = np.mean(np.array(mse_score_list))
    avg_ssim_score = np.mean(np.array(ssim_score_list))
    avg_psnr_score = np.mean(np.array(psnr_score_list))
    std_mse_score = np.std(np.array(mse_score_list))
    std_ssim_score = np.std(np.array(ssim_score_list))
    std_psnr_score = np.std(np.array(psnr_score_list))
    mi.logger.debug("== {} Training-based {} performance Score with optimizer {}, lr {}, loss type {} on {} epoch saved model ==".format(args.gan_AE_type, args.attack_scheme, args.MIA_optimizer, args.MIA_lr, args.attack_loss_type, args.num_epochs))
    
    if not args.attack_confidence_score:
        mi.logger.debug("Reverse Intermediate activation at layer {} (-1 is the smashed-data)".format(args.attack_from_later_layer))
    else:
        mi.logger.debug("Reverse Confidence Score, conventional MIA!")
    
    mi.logger.debug("MIA performance Score is (MSE, SSIM, PSNR) averaging {} times\n{}, {}, {}\n{}, {}, {}".format(len(random_seed_list), avg_mse_score, avg_ssim_score, avg_psnr_score, std_mse_score, std_ssim_score, std_psnr_score))

# %%
