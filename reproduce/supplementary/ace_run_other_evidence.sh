#!/bin/bash
cd "$(dirname "$0")"
# bash other_evidence/ace_run_expert_vgg11_cut2.sh
bash other_evidence/ace_run_finetune_lowLRlite_source_svhn.sh
bash other_evidence/ace_run_finetune_lowLRlite_target_cifar100.sh