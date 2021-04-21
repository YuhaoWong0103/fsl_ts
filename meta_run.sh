#!/bin/bash

# dir params
model_params_dir="./fsl_model_params/maml"
figure_dir="./fsl_figures/maml"
logs_dir="./fsl_logs/maml"
logs_name="./fsl_logs/maml/meta_info.log"
check_point="maml_init_params_epoch-5000_metalr-0.001_updatelr-0.1.pt"

# meta params
epoch=5000
n_way=3
k_spt=2
k_qry=8
task_num=8
task_num_eval=8
task_num_test=8
meta_lr=0.001
update_lr=0.1
update_step=10
update_step_test=10
clip_val=1.0

# cuda parmas
cuda_no=1

# dir check
if [ ! -d $logs_dir ];then
   mkdir -p $logs_dir
fi

CUDA_VISIBLE_DEVICES=$cuda_no python -u fsl_ts_maml.py --model_params_dir $model_params_dir \
                --figure_dir $figure_dir \
                --logs_dir $logs_dir \
                --epoch $epoch \
                --n_way $n_way \
                --k_spt $k_spt \
                --k_qry $k_qry \
                --task_num $task_num \
                --task_num_eval $task_num_eval \
                --task_num_test $task_num_test \
                --meta_lr $meta_lr \
                --update_lr $update_lr \
                --update_step $update_step \
                --update_step_test $update_step_test \
                --clip_val $clip_val \
                --check_point $check_point \
                --train \
                --pred \
                --eval \
                --test > $logs_name 2>&1 &