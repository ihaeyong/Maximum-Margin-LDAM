#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/LDAM-DRW
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == 'ce' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type CE \
           --seed 1 \
           --epochs 120 \
           --batch-size 128 \
           --train_rule None \
           --exp_str non_warmup_crop_adaptive_avg_pooling_non_crop

elif [ $1 == 'ldam' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type LDAM \
           --train_rule None \
           --scale 10.0 \
           --max_m 0.5\
           --seed 1 \
           --exp_str 0

elif [ $1 == 'hmm' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type  HMM \
           --train_rule DRW \
           --epochs 120 \
           --scale 10.0 \
           --max_m 2.7 \
           --gamma 1.2 \
           --seed 1 \
           --exp_str only

elif [ $1 == 'hmm-ldam' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type  HMM-LDAM \
           --train_rule DRW \
           --epochs 120 \
           --scale 10.0 \
           --max_m 2.7 \
           --gamma 1.1 \
           --seed 1 \
           --exp_str logits
fi
