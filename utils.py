import logging
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import math
import numbers
from torch.nn import functional as F
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
import matplotlib.pyplot as plt
import copy


def freeze_model_bn(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

def piecewise_clustering(var, lambda_coeff, l_norm):
    var1=(var[var.ge(0)]-var[var.ge(0)].mean()).pow(l_norm).sum()
    var2=(var[var.le(0)]-var[var.le(0)].mean()).pow(l_norm).sum()
    return lambda_coeff*(var1+var2)

def clustering_loss(model, lambda_coeff, l_norm=2):
    
    pc_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            pc_loss += piecewise_clustering(m.weight, lambda_coeff, l_norm)
    
    return pc_loss 

def prune_top_n_percent_left(tensor, n = 20):
    tensor_shape = tensor.size()
    num_ele = int(np.prod(tensor_shape) * (100 - n) / 100.)
    tensor = torch.flatten(tensor)
    _, index = tensor.topk(k = num_ele, dim = 0, largest = False)
    tensor[index] = 0.0
    return tensor.view(tensor_shape)


def dropout_defense(tensor, ratio = 0.5):
    tensor_shape = tensor.size()
    device = tensor.device
    A_array = torch.rand(tensor_shape) < ratio
    A_array = A_array.to(device)
    tensor = tensor.masked_fill(A_array, 0)
    return tensor

def prune_defense(tensor, ratio = 0.5):
    tensor_shape = tensor.size()
    num_ele = int(np.prod(tensor_shape) * ratio)
    tensor = torch.flatten(tensor)
    _, index = tensor.topk(k = num_ele, dim = 0, largest = False)
    tensor[index] = 0.0
    return tensor.view(tensor_shape)
# def spurious_score_V0(var, lambda_coeff, l_norm):
#     var1 = 1 / var.mean().pow(l_norm).sum()
#     return lambda_coeff*(var1)

# def spurious_loss_V0(model, lambda_coeff, l_norm=2):
    
#     pc_loss = 0
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             pc_loss += spurious_score(m.weight, lambda_coeff, l_norm)
    
#     return pc_loss 

def spurious_loss(act, lambda_coeff, l_norm = 2):
    if l_norm == 1:
        var1 = 20 / (torch.max(act) - torch.min(act)).pow(l_norm).sum()
    elif l_norm >= 2:
        var1 = 400 / (torch.max(act) - torch.min(act)).pow(l_norm).sum()
    return lambda_coeff*(var1)



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0].state_dict())
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i].state_dict()[key]
        w_avg[key] = torch.true_divide(w_avg[key], len(w))
    return w_avg

def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)

def save_grad(model):
    local_grad_stat_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            local_grad_stat_dict[name] = param.grad.detach().clone()
    return local_grad_stat_dict

def load_grad(model, state_dict, print_option = False):
    for name, param in model.named_parameters():
        if name in state_dict:
            param.grad = state_dict[name].detach().clone()
        else:
            print("missing key: ", name)
    if print_option:
        for name, param in model.named_parameters():
            print(param.grad)



def torch_diff(tensor_val1, tensor_val2):
    return tensor_val1 - tensor_val2, tensor_val1 > tensor_val2

def plot_change(change_list, save_dir):
    def plot_log(ax, x, y):
        ax.plot(x, y, color='black')
        ax.set(title="Loss Change through Training")
        ax.set_xlabel('Step', fontweight ='bold')
        ax.grid()

    fig1, ax = plt.subplots()
    x = np.arange(0, len(change_list)) * 50

    plot_log(ax, x, change_list)
    fig1.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

def accuracy(output, target, topk=(1,), compress_V4shadowlabel = False, num_client = 10):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if compress_V4shadowlabel:
        pred = pred % num_client
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_PSNR(refimg, invimg, peak = 1.0):
    psnr = 10*torch.log10(peak**2 / torch.mean((refimg - invimg)**2))
    return psnr

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
    	batch_size = int(source.size()[0])
    	kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
    	XX = kernels[:batch_size, :batch_size]
    	YY = kernels[batch_size:, batch_size:]
    	XY = kernels[:batch_size, batch_size:]
    	YX = kernels[batch_size:, :batch_size]
    	loss = torch.mean(XX + YY - XY -YX)
    	return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pairwise_dist_torch(A):
    sigma = torch.Tensor([1e-7]).to(A.device)
    r = torch.sum(A*A, axis = 1)
    r = r.view(-1, 1)
    # print(r)
    D = torch.maximum(r - 2*torch.matmul(A, A.t()) + r.t(), sigma)
    D = torch.sqrt(D)
    return D



class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def forward(self, input_data, intermediate_data):
        input_data = input_data.view(input_data.size(0), -1)
        intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

        # Get A matrices of data
        A_input = self._A_matrix(input_data)
        A_intermediate = self._A_matrix(intermediate_data)

        # Get distance variances
        input_dvar = self._distance_variance(A_input)
        intermediate_dvar = self._distance_variance(A_intermediate)

        # Get distance covariance
        dcov = self._distance_covariance(A_input, A_intermediate)

        # Put it together
        dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

        return dcorr

    def _distance_covariance(self, a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(self, a_matrix):
        return (a_matrix ** 2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(self, data):
        distance_matrix = self._distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(self, data):
        n = data.size(0)
        distance_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                row_diff = data[i] - data[j]
                distance_matrix[i, j] = (row_diff ** 2).sum()

        return distance_matrix

def dist_corr_torch(X, Y):



    n = float(X.size()[0])
    sigma = torch.Tensor([1e-7]).to(X.device)
    a = pairwise_dist_torch(X)
    b = pairwise_dist_torch(Y)
    # print(a, b)
    A = a - torch.mean(a, axis=1) - torch.unsqueeze(torch.mean(a, axis=0), axis=1) + torch.mean(a)
    B = b - torch.mean(b, axis=1) - torch.unsqueeze(torch.mean(b, axis=0), axis=1) + torch.mean(b)
    # print(A,B)
    dCovXY = torch.sqrt(sigma + torch.sum(A*B) / (n ** 2)) # Add sigma to avoid nan loss
    dVarXX = torch.sqrt(sigma + torch.sum(A*A) / (n ** 2))
    dVarYY = torch.sqrt(sigma + torch.sum(B*B) / (n ** 2))
    dCorXY = dCovXY / (torch.sqrt(sigma + dVarXX * dVarYY) + sigma)  # Add sigma to avoid nan loss
    return dCorXY

def dist_corr(img, act):
    flattened_input = img.view(img.size(0), -1)
    flattened_act = act.view(act.size(0), -1)
    return dist_corr_torch(flattened_input, flattened_act)

def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data

def deprocess(data, num_class = 10):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1 and num_class == 10:
        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    elif NChannels == 3 and num_class == 10:
        mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
        sigma = torch.tensor([0.247, 0.243, 0.261], dtype=torch.float32)
    elif NChannels == 3 and num_class == 100:
        mu = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=torch.float32)
        sigma = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=torch.float32)
    else:
        print("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def l2loss(x):
    return (x**2).mean()

def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
    return result

def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["weight_shape"] = list(module.weight.size())
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        # line_new = "{:>20}  {:>25} {:>15}".format(
        #     layer,
        #     str(summary[layer]["output_shape"]),
        #     "{0:,}".format(summary[layer]["nb_params"]),
        # )
        if "weight_shape" in summary[layer]:
            if len(summary[layer]["weight_shape"]) == 4:
                kernel_size = summary[layer]["weight_shape"][-1]
            else:
                kernel_size = "x"
        else:
            kernel_size = "x"

        line_new = "{:>20}  {:>25} {:>15}".format(
            layer + " ({}x{})".format(kernel_size, kernel_size),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)


class CustomPad(nn.Module):
  def __init__(self, padding):
    super(CustomPad, self).__init__()
    self.padding = padding
  def forward(self, input):
    return F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
    # F.pad(x. self.padding, mode='replicate')

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            # kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
            #           torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
class ImageTensorFolder(torch.utils.data.Dataset):

    def __init__(self, img_path, tensor_path, label_path = "None", img_fmt="npy", tns_fmt="npy", lbl_fmt="npy", transform=None, limited_num = None):
        self.img_fmt = img_fmt
        self.tns_fmt = tns_fmt
        self.lbl_fmt = lbl_fmt
        select_idx = None
        if limited_num is not None:
            limited_num_10 = (limited_num// 10) * 10
            select_idx = []
            visited_label = {}
            filepaths = label_path + "/*.{}".format(lbl_fmt)
            files = sorted(glob(filepaths))
            count = 0
            index = 0
            # for index in range(limited_num_10):
            while count < limited_num_10:
                label = self.load_tensor(files[index], file_format=self.lbl_fmt)
                if label.item() not in visited_label:
                    visited_label[label.item()] = 1
                    select_idx.append(index)
                    # print(label.item())
                    count += 1
                elif visited_label[label.item()] < limited_num_10 // 10:
                    visited_label[label.item()] += 1
                    select_idx.append(index)
                    # print(label.item())
                    count += 1
                index += 1
                # print(label.item())
        self.img_paths = self.get_all_files(img_path, file_format=img_fmt, select_idx = select_idx)
        self.tensor_paths = self.get_all_files(tensor_path, file_format=tns_fmt, select_idx = select_idx)
        if label_path != "None":
            self.label_paths = self.get_all_files(label_path, file_format=lbl_fmt, select_idx = select_idx)
        else:
            self.label_paths = None
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def get_all_files(self, path, file_format="png", select_idx = None):
        filepaths = path + "/*.{}".format(file_format)
        files = sorted(glob(filepaths))
        # print(files[0:10])
        if select_idx is None:
            return files
        else:
            file_list = []
            for i in select_idx:
                file_list.append(files[i])
            return file_list

    def load_img(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            img = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(img).shape[0] == 4:
                img = self.to_tensor(img)[:3, :, :]
                img = self.to_pil(img)
        elif file_format == "npy":
            img = np.load(filepath)
            #cifar10_mean = [0.4914, 0.4822, 0.4466]
            #cifar10_std = [0.247, 0.243, 0.261]
            img = np.uint8(255 * img)
            img = self.to_pil(img)
        elif file_format == "pt":
            img = torch.load(filepath)
        else:
            print("Unknown format")
            exit()
        return img

    def load_tensor(self, filepath, file_format="png"):
        if file_format == "png":
            tensor = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(tensor).shape[0] == 4:
                tensor = self.to_tensor(tensor)[:3, :, :]
        elif file_format == "npy":
            tensor = np.load(filepath)
            tensor = self.to_tensor(tensor)
        elif file_format == "pt":
            tensor = torch.load(filepath)
            if len(tensor.size()) == 4:
                tensor = tensor.view(tensor.size()[1:])
            # print(tensor.size())
            tensor.requires_grad = False
        elif file_format == "label":
            tensor = torch.load(filepath)
            if len(tensor.size()) == 4:
                tensor = tensor.view(tensor.size()[1:])
            # print(tensor.size())
            tensor.requires_grad = False
        return tensor

    def __getitem__(self, index):
        img = self.load_img(self.img_paths[index], file_format=self.img_fmt)

        if self.transform is not None:
            img = self.transform(img)

        intermed_rep = self.load_tensor(self.tensor_paths[index], file_format=self.tns_fmt)

        if self.label_paths is not None:
            label = self.load_tensor(self.label_paths[index], file_format=self.lbl_fmt)
            return img, intermed_rep, label
        else:
            return img, intermed_rep

    def __len__(self):
        return len(self.img_paths)

from torch.utils.data import SubsetRandomSampler
def apply_transform_test(batch_size, image_data_dir, tensor_data_dir, limited_num = None, shuffle_seed = 123, dataset = None):
    """
    """
    std = [1.0, 1.0, 1.0]
    mean = [0.0, 0.0, 0.0]
    # if dataset is None:
    #     std = [1.0, 1.0, 1.0]
    #     mean = [0.0, 0.0, 0.0]
    # elif dataset == "cifar10":
    #     std = [0.247, 0.243, 0.261]
    #     mean = [0.4914, 0.4822, 0.4465]
    # elif dataset == "cifar100":
    #     std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    #     mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    # elif dataset == "imagenet":
    #     std = [0.229, 0.224, 0.225]
    #     mean = [0.485, 0.456, 0.406]
    # elif dataset == "facescrub":
    #     std = [0.5, 0.5, 0.5]
    #     mean = [0.5, 0.5, 0.5]

    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])
    dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir, label_path=tensor_data_dir,
                                 img_fmt="jpg", tns_fmt="pt", lbl_fmt="label", transform=trainTransform, limited_num = limited_num)
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # np.random.seed(shuffle_seed)
    # np.random.shuffle(indices)
    # test_indices = indices[0:]
    # test_sampler = SubsetRandomSampler(test_indices)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return testloader

def apply_transform(batch_size, image_data_dir, tensor_data_dir, shuffle_seed = 123, dataset = None):
    """
    """
    std = [1.0, 1.0, 1.0]
    mean = [0.0, 0.0, 0.0]
    # if dataset is None:
    #     std = [1.0, 1.0, 1.0]
    #     mean = [0.0, 0.0, 0.0]
    # elif dataset == "cifar10":
    #     std = [0.247, 0.243, 0.261]
    #     mean = [0.4914, 0.4822, 0.4465]
    # elif dataset == "cifar100":
    #     std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    #     mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    # elif dataset == "imagenet":
    #     std = [0.229, 0.224, 0.225]
    #     mean = [0.485, 0.456, 0.406]
    # elif dataset == "facescrub":
    #     std = [0.5, 0.5, 0.5]
    #     mean = [0.5, 0.5, 0.5]
    
    train_split = 0.9
    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])
    dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir,
                                 img_fmt="jpg", tns_fmt="pt", transform=trainTransform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.seed(shuffle_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             sampler=test_sampler)
    return trainloader, testloader

# if __name__ == "__main__":
#     pred = torch.randn((4, 5))
#     print(pred)
#     pruned_pred = dropout_defense(pred, 0.8)
#     print(pruned_pred)