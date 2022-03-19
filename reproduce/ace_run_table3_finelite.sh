#!/bin/bash
cd "$(dirname "$0")"
# bash table3/ace_run_expert_vgg11_cut2.sh
bash table3/ace_run_finetune_lowLRlite_source_cifar10.sh
bash table3/ace_run_finetune_lowLRlite_source_cifar100.sh
# bash table3/ace_run_finetune_freeze_source_cifar10.sh
# bash table3/ace_run_finetune_freeze_source_cifar100.sh
# bash table3/ace_run_finetune_lowLRsimple_source_cifar10.sh
# bash table3/ace_run_finetune_lowLRsimple_source_cifar100.sh