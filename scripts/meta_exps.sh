#!/bin/bash

source user_config.sh
seed=1491

# debug
python main.py --config config/meta.conf --overrides "exp_name = metalearn, run_name = debug-meta, max_vals = 3, val_interval = 100, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 1, sent_enc = conv, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, approx_term = sign_cos_sim"

# profile
#python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-exact-nograph, max_vals = 1, val_interval = 10, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 0, sent_enc = conv, d_hid = 128, mnli_pair_attn = 0, snli_pair_attn = 0, batch_size = 4, val_data_limit = 100"
#python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-approx, max_vals = 1, val_interval = 10, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, metatrain = 1, slow_params_approx = 1, sent_enc = conv, d_hid = 256, mnli_pair_attn = 0, snli_pair_attn = 0, val_data_limit = 100"

## FEEL THE METALEARN ##
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = exact-mnli-snli-schdcos-rs, slow_params_approx = 0, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00001, random_seed = ${seed}, scheduler = cosine" 
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = approx-mnli-snli-schdcos-r2, slow_params_approx = 1, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00001, random_seed = 222, scheduler = cosine" 
