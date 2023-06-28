model_name=DySAT
dataset=reddit_hyperlinks #radoslaw #auton-sys
task=link_prediction

method=degree

num_graphs=20
context=19
target_ts=19
ntargets=1
nsamples=100
khop=2

constraint=noise

online=false

tga_disturbance_ratio=0
tga_thresh_prob=0.5

declare -a epsilons_arr=(0.02 0.06 0.1 0.3 0.5 0.7 0.9)

epsilon1=100

# 2 3 5 7 4, 6

if [ "$constraint" == noise_feat ]; then
    file=baseline_main_feat.py
else
    file=baseline_main.py
fi

seed=123 #123 483, 665, 811

saved_model=models/${model_name}/${dataset}
resparent_dir="baselines/${method}/results_${model_name,,}"
if [ ! -d "${resparent_dir}/${dataset}/multi_targets/${constraint}" ]; then
    mkdir -p ${resparent_dir}/${dataset}/multi_targets/${constraint}
fi
results_dir="baselines/${method}/results"
if [ ! -d "${results_dir}/${dataset}/${constraint}" ]; then
    mkdir -p ${results_dir}/${dataset}/${constraint}
fi
for epsilon in "${epsilons_arr[@]}"
do
    python3 $file \
        -method ${method} \
        -model_name ${model_name} \
        -constraint ${constraint} \
        -budget ${budget} \
        -epsilon ${epsilon} \
        -epsilon1 ${epsilon1} \
        -saved_model ${saved_model} \
        -dataset ${dataset} \
        -task ${task} \
        -num_graphs ${num_graphs} \
        -ntargets ${ntargets} \
        -khop ${khop} \
        -context ${context} \
        -target_ts ${target_ts} \
        -featureless \
        -num_samples $nsamples \
        -device cpu \
        -neg_sampling \
        -tga_thresh_prob $tga_thresh_prob \
        -tga_disturbance_ratio $tga_disturbance_ratio \
        -tga_iter_ub 500 \
        -seed ${seed} \
        -sampling rd \
        -analyze_only > ${resparent_dir}/${dataset}/multi_targets/${constraint}/results_td_tg${ntargets}_n${num_graphs}_c${context}t${target_ts}_e${epsilon}_eb${epsilon1}_seed${seed}.txt
        # -dyn_feats \
        # _seed${seed}.txt
        # -large_graph \
        # -debug \
        # 
        # b${budget}_l${lambda1}.txt 
        # -seq_tg_attk \
done