## [Learning imbalanced datasets with maximum margin loss](https://arxiv.org/abs/2206.05380)
Kang, Haeyong and Vu, Thang and Yoo, Chang D
_________________

This is the official implementation of MM-LDAM-DRW in the paper [Learning Imbalanced Datasets with Maximum Margin Loss](https://arxiv.org/abs/2206.05380) in PyTorch.


Our baseline code follows the official implementation of LDAM-DRW in the paper [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/pdf/1906.07413.pdf) in PyTorch.

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.11.0
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [scikit-learn](https://scikit-learn.org/stable/)

### Dataset

- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- The paper also reports results on Tiny ImageNet and iNaturalist 2018. We will update the code for those datasets later.

### Training 

We provide several training examples with this repo:

- To train the ERM baseline on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```

- To train the LDAM Loss along with DRW training on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
```


- To train the **Maximum Margin Loss along with DRW training** on long-tailed imbalance with ratio of 100 (hyper-params of scale, max_m, and gamma should be set)

```bash
python cifar_train.py --dataset cifar100 --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type HMM \
           --train_rule DRW --epochs 200 --scale 19.0 --max_m 1.29 --gamma 1.528 --seed 1 --exp_str logits
```


### Reference

If you find our paper and repo useful, please cite as
```
@inproceedings{kang2021learning,
  title={Learning imbalanced datasets with maximum margin loss},
  author={Kang, Haeyong and Vu, Thang and Yoo, Chang D},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={1269--1273},
  year={2021},
  organization={IEEE}
}
```
