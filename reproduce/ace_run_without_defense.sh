#!/bin/bash
cd "$(dirname "$0")"
bash without_defense/ace_run_vgg11_baseline_cifar100.sh
bash without_defense/ace_run_resnet_baseline_cifar100.sh
bash without_defense/ace_run_mobilenet_baseline_cifar100.sh