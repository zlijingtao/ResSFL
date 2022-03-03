from architectures_torch import *
from vgg import vgg11_bn
from resnet_cifar import ResNet20

'''Select Model below'''

# resnet_cut3 = ResNet20(3, None, num_class = 10)
vgg_cut2 = vgg11_bn(4, None, num_class = 10)
vgg_cut2.cuda()
smashed_data_size = vgg_cut2.get_smashed_data_size()
print(vgg_cut2)
print("Smashed data size is", smashed_data_size)
'''Select AE model below'''


# AE_model = conv_normN_AE(N = 0, internal_nc = 16, input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()
AE_model = res_normN_AE(N = 4, internal_nc = 64, input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()

'''Print out Model Summary'''
print(AE_model)
# summary(AE_model, smashed_data_size[1:])