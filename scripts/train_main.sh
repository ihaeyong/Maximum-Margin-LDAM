#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/LDAM-DRW
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == 'ce' ]; then
    python cifar_train.py \
           --dataset cifar100 \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type CE \
           --train_rule None

elif [ $1 == 'ldam' ]; then
    python cifar_train.py \
           --dataset cifar100 \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type LDAM \
           --train_rule DRW \
           --seed 1 \
           --exp_str logits

elif [ $1 == 'hmm' ]; then
    python cifar_train.py \
           --dataset cifar100 \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type HMM  \
           --train_rule DRW \
           --epochs 200 \
           --scale 19.0 \
           --max_m 1.29 \
           --gamma 1.528 \
           --seed 1\
           --exp_str hmm

elif [ $1 == 'hmm-ldam' ]; then
    python cifar_train.py \
           --dataset cifar100 \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type HMM-LDAM \
           --train_rule DRW \
           --epochs 200 \
           --scale 26.0 \
           --max_m 1.29 \
           --gamma 1.528 \
           --seed 1 \
           --exp_str hmm-ldam

fi
