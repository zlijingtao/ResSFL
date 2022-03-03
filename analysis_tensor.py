import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np  
   


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--folder_name', required=True, type=str, help='please type folder_name name for the testing purpose')
parser.add_argument('--file_name', required=True, type=str, help='please type save_file name for the testing purpose')
args = parser.parse_args()

tensor_path = "./" + args.folder_name + "/"+ args.file_name + "/saved_tensors"
print(tensor_path)
analysis_path = "./" + args.folder_name + "/"+ args.file_name + "/tensor_analysis"
if not os.path.isdir(tensor_path):
    raise("Tensors not found!")
if not os.path.isdir(analysis_path):
    os.makedirs(analysis_path)


params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
        }
rcParams.update(params)
for filename in os.listdir(tensor_path):
    if filename.endswith(".npy"):
        npy_file_path = os.path.join(tensor_path, filename)
        np_array = np.load(npy_file_path)
        if "client" in filename and "Sequential" in filename:
            print("mean is {}".format(np.mean(np_array)))
            print("max is {}".format(np.max(np_array)))
            print("min is {}".format(np.min(np_array)))
            print("std is {}".format(np.std(np_array)))

        hist, bins = np.histogram(np_array, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        # hist = np.histogram(np_array, bins=10, range=None, normed=None, weights=None, density=None)
        # plt.plot(hist) 
        # plt.hist(a, bins = [0,20,40,60,80,100]) 
        plt.title("histogram")
        
        # plt.savefig(analysis_path+'/{}.pdf'.format(filename.split(".")[0]), bbox_inches='tight')
        plt.savefig(analysis_path+'/{}.png'.format(filename.split(".")[0]))
        plt.clf()
    else:
        continue