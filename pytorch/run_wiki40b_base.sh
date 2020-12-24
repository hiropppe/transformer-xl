#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/wiki-40b/$2 \
        --datase sp \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 60 \
        --multi_gpu \
        --spm_file ../data/wiki-40b/$2/m.model
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/example/ \
        --dataset sp \
        --work_dir $2 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        --spm_file ../data/wiki-40b/$2/m.model
else
    echo 'unknown argment 1'
fi
