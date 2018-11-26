#!/bin/bash

source user_config.sh

exp_name="metalearn"
run_name="debug-tb"
clear_ckpts=0
if [ ${clear_ckpts} -eq 1 ]
then
    exp_dir="${JIANT_PROJECT_PREFIX}/${exp_name}/${run_name}"
    echo "Removing model files from ${exp_dir} in 5 seconds."
    sleep 5
    rm -rf ${exp_dir}/*.th 
fi

# debug
#python main.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = debug-exact, max_vals = 1, val_interval = 10, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 0, sent_enc = conv, d_hid = 256, mnli_pair_attn = 0, snli_pair_attn = 0, batch_size= 8"

# profile
python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-exact, max_vals = 1, val_interval = 10, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 0, sent_enc = conv, d_hid = 128, mnli_pair_attn = 0, snli_pair_attn = 0, batch_size = 4, val_data_limit = 100"
#python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-approx, max_vals = 1, val_interval = 10, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 1, sent_enc = conv, d_hid = 256, mnli_pair_attn = 0, snli_pair_attn = 0, val_data_limit = 100"

## FEEL THE METALEARN ##
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = glue-meta-attn-pfinetune-r1, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, cuda = 0" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = glue-meta-attn-pfinetune-r2, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, random_seed = 5678, cuda = 0" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = glue-meta-attn-pfinetune-r3, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, random_seed = 91011, cuda = 1" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", eval_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = glue-meta-noelmo-pfinetune-r3, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, random_seed = 91011, cuda = 0" 

## Train on a subset of GLUE, test on all

# With attention
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-meta-attn-pfinetune-r1, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, cuda = 1" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-meta-attn-pfinetune-r2, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, random_seed = 5678, cuda = 2" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-meta-attn-pfinetune-r3, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 0, train_for_eval = 1, do_eval = 1, random_seed = 91011, cuda = 2" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-nometa-attn-nofinetune-r1, elmo_chars_only = 1, metatrain = 0, allow_reuse_of_pretraining_parameters = 0, do_train = 1, train_for_eval = 0, do_eval = 1, cuda = 5" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-nometa-attn-nofinetune-r2, elmo_chars_only = 1, metatrain = 0, allow_reuse_of_pretraining_parameters = 0, do_train = 1, train_for_eval = 0, do_eval = 1, random_seed = 5678, cuda = 0" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli,mrpc,sts-b,rte,cola\", eval_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 5000, run_name = gluesubset-nometa-attn-nofinetune-r3, elmo_chars_only = 1, metatrain = 0, allow_reuse_of_pretraining_parameters = 0, do_train = 1, train_for_eval = 0, do_eval = 1, random_seed = 91011, cuda = 1" 

# Without attention
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli-alt,mrpc,sts-b-alt,rte,cola\", eval_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 5000, run_name = gluesubset-meta-nofinetune-r0, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 1, train_for_eval = 0, do_eval = 1, random_seed = 111, cuda = 0" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli-alt,mrpc,sts-b-alt,rte,cola\", eval_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 5000, run_name = gluesubset-meta-nofinetune-r1, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 1, train_for_eval = 0, do_eval = 1, random_seed = 222, cuda = 0" 
#python main.py --config config/final.conf --overrides "train_tasks = \"mnli-alt,mrpc,sts-b-alt,rte,cola\", eval_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 5000, run_name = gluesubset-meta-nofinetune-r2, elmo_chars_only = 1, metatrain = 1, allow_reuse_of_pretraining_parameters = 1, do_train = 1, train_for_eval = 0, do_eval = 1, random_seed = 333, cuda = 0" 


