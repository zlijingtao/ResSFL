#!/bin/bash
cd "$(dirname "$0")"
bash table2/ace_run_finetune_lowLRlite_MA.sh
bash table2/ace_run_finetune_lowLRlite_target_cifar100_100clients_cifar100.sh
bash table2/ace_run_finetune_lowLRlite_target_cifar100_100clients_svhn.sh