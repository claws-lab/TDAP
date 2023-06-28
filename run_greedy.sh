model_name=DySAT
dataset=reddit_hyperlinks #radoslaw #auton-sys
task=link_prediction

saved_model=models/${model_name}/${dataset}

num_graphs=20
context=19
target_ts=19
ntargets=1
nsamples=100
khop=2

constraint=noise

epsilon=0.1
budget=10

online=false

tga_disturbance_ratio=0
tga_thresh_prob=0.5

declare -a epsilons_arr=(0.02 0.06 0.1 0.3 0.5 0.7 0.9) 

epsilon1=100

resparent_dir="baselines/greedy/results_${model_name,,}"

if [ ! -d "${resparent_dir}/${dataset}/multi_targets/${constraint}" ]; then
    mkdir -p ${resparent_dir}/${dataset}/multi_targets/${constraint}
fi
file=greedy_main.py
echo $file

seed=123 #123 483, 665, 811

for epsilon in "${epsilons_arr[@]}"
do
    python3 $file \
        -constraint ${constraint} \
        -budget ${budget} \
        -epsilon ${epsilon} \
        -epsilon1 ${epsilon1} \
        -model_name ${model_name} \
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
        -device cuda:6 \
        -neg_sampling \
        -tga_thresh_prob $tga_thresh_prob \
        -tga_disturbance_ratio $tga_disturbance_ratio \
        -tga_iter_ub 500 \
        -seed ${seed} > ${resparent_dir}/${dataset}/multi_targets/${constraint}/results_tg${ntargets}_n${num_graphs}_c${context}t${target_ts}_e${epsilon}_eb${epsilon1}_l${lambda1}.txt \ #_seed${seed}.txt
        # -sampling td \
        # -dyn_feats \
        # _seed${seed}.txt
        # -large_graph \
        # -debug \
done
