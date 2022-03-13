#!/bin/bash
cd "$(dirname "$0")"
bash table6/ace_run_expert_vgg11_cut1.sh
bash table6/ace_run_expert_vgg11_cut2.sh
bash table6/ace_run_expert_vgg11_cut3.sh
bash table6/ace_run_finetune_lowLRlite_vgg.sh

bash table6/ace_run_expert_resnet20_cut2.sh
bash table6/ace_run_expert_resnet20_cut3.sh
bash table6/ace_run_expert_resnet20_cut4.sh
bash table6/ace_run_finetune_lowLRlite_resnet.sh

bash table6/ace_run_expert_mobilenet_cut2.sh
bash table6/ace_run_expert_mobilenet_cut3.sh
bash table6/ace_run_expert_mobilenet_cut4.sh
bash table6/ace_run_finetune_lowLRlite_mobilenetv2.sh