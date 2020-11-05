#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data ../data/example/ \
        --datase sp \
        --n_layer 4 \
        --d_model 210 \
        --n_head 5 \
        --d_head 20 \
        --d_inner 1000 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 100 \
        --mem_len 100 \
        --eval_tgt_len 100 \
        --batch_size 16 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --data ../data/example/ \
        --dataset sp \
        --work_dir $2 \
        --tgt_len 100 \
        --mem_len 100 \
        --clamp_len -1 \
        --same_length \
        --split test \
        --spm_file ../data/example/m.model \
        ${@:3}
else
    echo 'unknown argment 1'
fi
