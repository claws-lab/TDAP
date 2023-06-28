#!/bin/bash
#! /bin/bash

task=node_classification
dataset=reddit #opsahl-ucsocial #auton-sys # facebook-wosn
num_graphs=10 #11 #14
context=9 #10 #13
target_ts=9 #10 #13

model_name=DySAT

nepochs=500
learning_rate=0.0001
neg_weight=0.001
neg_sample_size=100
batch_size=128
min_time_perc=0.6

device=cuda:6

if [ "$task" == node_classification ]; then
    file=train_models_nc.py
else
    file=train_models.py
fi

python3 $file \
    -task $task \
    -dataset $dataset \
    -num_graphs $num_graphs \
    -context $context \
    -target_ts $target_ts \
    -model_name $model_name \
    -emb_size 32 \
    -decoder_sizes 32 32 \
    -chebyK 5 \
    -dys_struc_head 16 8 8 \
    -dys_struc_layer 32 \
    -dys_temp_head 16 \
    -dys_temp_layer 32 \
    -dys_spa_drop 0.1 \
    -dys_temp_drop 0.5 \
    -nepochs $nepochs \
    -learning_rate $learning_rate \
    -neg_weight $neg_weight \
    -neg_sample_size $neg_sample_size \
    -batch_size $batch_size \
    -min_time_perc $min_time_perc \
    -device $device \
    -data_sample \
    -sample_prop 0.1 \
    -seed 123 \
    -patience 10 \
    -dyn_feats \
    -logging  \
    # -reg_lambda 0.0001 \
    # -to_save \
    # -featureless \
