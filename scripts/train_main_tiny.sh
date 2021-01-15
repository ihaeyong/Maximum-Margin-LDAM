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
           --exp_str warmup_1crop

elif [ $1 == 'ldam' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type LDAM \
           --train_rule DRW \
           --seed 1 

elif [ $1 == 'unbiased' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type Unbiased \
           --train_rule Unbiased \
           --epochs 200 \
           --scale 10.0 \
           --max_m 0.8 \
           --gamma 1.0 \
           --seed 1 \
           --exp_str const_margin

elif [ $1 == 'unbiased-ldam' ]; then
    python tiny_imagenet_train.py \
           --dataset tiny \
           --gpu 0 --imb_type exp \
           --imb_factor 0.01 \
           --loss_type Unbiased-ldam \
           --train_rule Unbiased-ldam \
           --epochs 120 \
           --scale 30.0 \
           --max_m 2.6 \
           --gamma 1.1 \
           --seed 1 \
           --exp_str minus
fi
