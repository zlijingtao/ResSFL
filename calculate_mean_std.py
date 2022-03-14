from datasets_torch import get_facescrub_bothloader, get_SVHN_trainloader, get_tinyimagenet_bothloader


''' Simple utility script to derive channel-wise mean and std of the raw training dataset (after scaling it to (0, 1))'''

'''WARNING: need to remove trasform operations except for transforms.ToTensor(), i.e. comment line 240, 241 and 243. leave line 242 which is ToTensor().'''

''' Remember to uncommented those after you derive the mean and std'''

# client_dataloader, pub_dataloader = get_facescrub_bothloader(batch_size=39969, 
#                                                                                 num_workers=4,
#                                                                                 shuffle=True,
#                                                                                 num_client=1,
#                                                                                 collude_use_public=False)
                                                                                
# data = next(iter(client_dataloader[0]))
# print(data[0][0, :, ].mean(), data[0][0, :, ].std())
# print(data[0][1, :, ].mean(), data[0][1, :, ].std())
# print(data[0][2, :, ].mean(), data[0][2, :, ].std())

# client_dataloader, pub_dataloader = get_tinyimagenet_bothloader(batch_size=100000, 
#                                                                                 num_workers=4,
#                                                                                 shuffle=True,
#                                                                                 num_client=1,
#                                                                                 collude_use_public=False)
                                                                                
# data = next(iter(client_dataloader[0]))
# print(data[0][0, :, ].mean(), data[0][0, :, ].std())
# print(data[0][1, :, ].mean(), data[0][1, :, ].std())
# print(data[0][2, :, ].mean(), data[0][2, :, ].std())

# client_dataloader, _, _ = get_SVHN_trainloader(batch_size=73257, num_workers=4,shuffle=True, num_client=1,collude_use_public=False)
# iterator_0 = iter(client_dataloader[0])                                         
# data = next(iterator_0)
# print(data[0][:, 0, ])
# print(data[0][0, :, ].mean(), data[0][0, :, ].std())
# print(data[0][1, :, ].mean(), data[0][1, :, ].std())
# print(data[0][2, :, ].mean(), data[0][2, :, ].std())