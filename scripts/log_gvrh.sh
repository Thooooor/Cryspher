#!/bin/bash
python main.py \
    --model graph_transformer \
    --layers 3 \
    --dim 128 \
    --init_atom_dim 92 \
    --init_edge_dim 1 \
    --cuda True \
    --device 1 \
    --epochs 500 \
    --patience 10 \
    --eval_freq 5 \
    --batch_size 128 \
    --optimizer Adam \
    --lr 0.001 \
    --weight_decay 0.001 \
    --momentum 0.95 \
    --data_dir ./datasets/log_gvrh/ \
    --data_file matbench_log_gvrh.json \
    --dataset log_gvrh \
    --train_ratio 0.6 \
    --valid_ratio 0.2 \
    --test_ratio 0.2 \