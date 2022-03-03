# MIA Defense: ResSFL framework: Attacker-Aware Training + Bottleneck Layers -> Resistance Transfer.
This repository is to reproduce ResSFL framework for defending MIA in split learning.

## Requirement:
tensorboard>1.15

pytorch

torchvision

thop (pip install)


## Code:

* *MIA_torch.py*: It implements the all utility functions of split learning,running MIA attacks and perform various defenses.
* *main_MIA.py*: Entry code to train a defensive model/vanilla model.
* *main_test_MIA.py*: Entry code to resume a trained model and perform MIA attack.

## Proof Of Concepts:
Run scripts directly in ./reproduce/ to reproduce main results in the manuscript. You must have a GPU with >= 8 GB memory.

For example:

```
bash reproduce/ace_run_table3.sh
```