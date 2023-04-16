#!/bin/bash
python train_sat.py \
    --model graph_transformer \
    --layers 6 \
    --dim 64 \
    --init_atom_dim 92 \
    --init_edge_dim 1 \
    --cuda True \
    --device 1 \
    --epochs 2000 \
    --patience 10 \
    --eval_freq 1 \
    --batch_size 128 \
    --optimizer Adam \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --momentum 0.95 \
    --warm_up 5000 \
    --data_dir ./datasets/e_form/ \
    --data_file matbench_mp_e_form.json \
    --dataset e_form \
    --subset True \
    --train_ratio 0.6 \
    --valid_ratio 0.2 \
    --test_ratio 0.2 \