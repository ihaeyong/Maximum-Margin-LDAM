#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/LDAM-DRW
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == 'ce' ]; then
    python cifar_train.py \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type CE \
           --train_rule None

elif [ $1 == 'ldam' ]; then
    python cifar_train.py \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type LDAM \
           --train_rule DRW

elif [ $1 == 'unbiased' ]; then
    python cifar_train.py \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type Unbiased \
           --train_rule None \
           --epochs 300 \
           --skew_th 2.9 \
           --ent_sc 1.0 \
           --exp_str unbiased

elif [ $1 == 'unbiased-ldam' ]; then
    python cifar_train.py \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type Unbiased-ldam \
           --train_rule Unbiased \
           --epochs 200 \
           --skew_th 2.9 \
           --ent_sc 1.0 \
           --scale 30.0 \
           --max_m 0.02 \
           --gamma 2.0 \
           --exp_str unbiased-ldam-origin

fi
