model_name=DySAT
dataset=reddit_hyperlinks #auton-sys #opsahl-ucsocial #radoslaw #auton-sys
task=link_prediction #link_prediction

saved_model=models/${model_name}/${dataset}

num_graphs=20 #13 #14
context=19 #12 #13
target_ts=19 #12 #13
ntargets=1
nsamples=50 #100

method=pgd
constraint=noise

num_steps=50
lambda1=0
khop=2
nprocs=1

lr_init=10

online=false

if $online; then
    file=src/main_online.py
elif [ "$constraint" == noise_feat ]; then
    file=src/main_feat.py
else
    file=src/main.py
fi

declare -a epsilons_arr=(0.02 0.06 0.1 0.3 0.5 0.7 0.9) 

epsilon1=100

resparent_dir="results_${model_name,,}"

if [ ! -d "${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}" ]; then
    mkdir -p ${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}
fi

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
        -num_steps ${num_steps} \
        -lambda1 ${lambda1} \
        -context ${context} \
        -target_ts ${target_ts} \
        -method ${method} \
        -featureless \
        -num_samples $nsamples \
        -device cuda:5 \
        -pll_devices cuda:0 cuda:2 cuda:3 cuda:5  \
        -neg_sampling \
        -seed ${seed} \
        -lr_init ${lr_init} \
        -nprocs $nprocs \
        -use_optim \
        -sampling None \
        -inits ones > ${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}/results_td_tg${ntargets}_n${num_graphs}_c${context}t${target_ts}_e${epsilon}_eb${epsilon1}_l${lambda1}_seed${seed}.txt
        # -dyn_feats \
        # b${budget}_l${lambda1}.txt 
        # -seq_tg_attk \
        # -seq_order deg_desc \
done
