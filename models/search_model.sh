#!/bin/bash
#! /bin/bash

task=node_classification
dataset=ethereum_phishing
num_graphs=10 #20
context=9 #19
target_ts=9 #19

model_name=DySAT

batch_size=128
learning_rate=0.01

neg_sample_size=10
neg_weight=0.0001
min_time_perc=0.9
# Parallelize
# cuda:1: 50, 0.005, 0.8, 0.01
neg_sample_sizes=(10 50 100 500 1000) #20 200 1000
neg_weights=(0.0001 0.001 0.01 0.1) #0.5
# min_time_percs=(0.6 0.7 0.8 0.9)
nepochs=(100 500) # 1000)
learning_rates=(0.0001 0.001 0.01 0.1)
# chebyKs=(2 3 4 5 6)

device=cuda:3
devices=(cuda:3) # cuda:2 cuda:3) #  cuda:1 cuda:3 cuda:2 cuda:5) # cuda:2)
tot_ndevs=${#devices[@]}
ndevice=0

for nepochs in "${nepochs[@]}"; do
    for neg_sample_size in "${neg_sample_sizes[@]}"; do
        for neg_weight in "${neg_weights[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                device=${devices[${ndevice}]}
                echo "${device}: $nepochs, $neg_sample_size, $neg_weight, $learning_rate"
                # echo "${device}: $neg_sample_size, $neg_weight, $min_time_perc, $learning_rate"
                # echo "${device}: $nepochs, $neg_sample, $learning_rate"
                python3 train_models_nc.py \
                    -task $task \
                    -dataset $dataset \
                    -num_graphs $num_graphs \
                    -context $context \
                    -target_ts $target_ts \
                    -featureless \
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
                    -seed 123 \
                    -patience 20 \
                    -dyn_feats \
                    -device $device >> new_${model_name,,}_${dataset}_${task}.txt
                    echo -e "-------" >> new_${model_name,,}_${dataset}_${task}.txt
                    # -data_sample \
                    # -sample_prop 0.1 \
                # ndevice=$(( (ndevice+1)%tot_ndevs ))
            done
        done
    done
done

# sem --wait