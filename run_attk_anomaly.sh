model_name=DySAT
dataset=reddit_hyperlinks #auton-sys #opsahl-ucsocial #radoslaw #auton-sys
task=link_prediction #link_prediction

saved_model=models/${model_name}/${dataset}

num_graphs=20 #13 #14
context=19 #12 #13
target_ts=19 #12 #13
ntargets=1
nsamples=100

method=pgd
constraint=noise #noise

num_steps=50
lambda1=0
khop=2
nprocs=1

epsilon=0.1
budget=10

lr_init=10

online=false

declare -a epsilons_arr=(0.02 0.06 0.1 0.3 0.5 0.7 0.9)

# epsilon1=$((ntargets * 2))
epsilon1=100

kanom=2

# 2 3 5 7 4, 6
resparent_dir="results_${model_name,,}_anom"

if [ ! -d "${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}" ]; then
    mkdir -p ${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}
fi

seed=123 #123 483, 665, 811

# for budget in "${budgets_arr[@]}"
for epsilon in "${epsilons_arr[@]}"
do
    python3 main_anomaly.py \
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
        -device cuda:1 \
        -pll_devices cuda:0 cuda:2 cuda:3 cuda:5  \
        -neg_sampling \
        -seed ${seed} \
        -lr_init ${lr_init} \
        -nprocs $nprocs \
        -use_optim \
        -sampling rd \
        -k_anom ${kanom} \
        -inits ones > ${resparent_dir}/${dataset}/multi_targets/${method}/${constraint}/results${kanom}_tg${ntargets}_n${num_graphs}_c${context}t${target_ts}_e${epsilon}_eb${epsilon1}_l${lambda1}_seed${seed}.txt
        # -dyn_feats \
        # b${budget}_l${lambda1}.txt 
        # -seq_tg_attk \
        # -seq_order deg_desc \
        #  
        # -large_graph \
done

# cu:1, 2, 3, 4, 5, 6, 7
