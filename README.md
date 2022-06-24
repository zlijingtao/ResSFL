# ResSFL framework - improving SFL's resistance to model inversion attack
Official Repository for ResSFL (resistant split federated learning)

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
Run scripts directly in ./reproduce/ to reproduce main results in the manuscript. You must have a GPU (CUDA-enabled) with >= 8 GB memory.

Recommend sequence to execute:
```
bash reproduce/ace_run_without_defense.sh
bash reproduce/ace_run_table6.sh
```

Comparison table:
```
bash reproduce/ace_run_table4.sh
```

Multi-client performance (with sampling, scale to 100 clients):
```
bash reproduce/ace_run_table2.sh
bash reproduce/ace_run_table2_sample.sh
```



List of additional experiments in supplementary material:

1. Performance against Optimization-based MI attack
2. Performance on clients with Non-iid data
3. Extra empirical evidence on successful resistance transfer

## Cite the work:
```
@inproceedings{li2022ressfl,
  title={ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning},
  author={Li, Jingtao and Rakin, Adnan Siraj and Chen, Xing and He, Zhezhi and Fan, Deliang and Chakrabarti, Chaitali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10194--10202},
  year={2022}
}
```
