#!/bin/bash

source user_config.sh
seed=222
gpuid=7

SMALLLM_CONVS='[(512, 3)] * 20'
BASELM_CONVS='[(1268, 4)] * 13'
MINIDAUPHIN_WIKI103_CONVS='[(850, 6)] * 3 + [(850, 1)] * 1 + [(850, 5)] * 4 + [(850, 1)] * 1 + [(850, 4)] * 3 + [(1024, 4)] * 1 + [(1024, 4)] * 1'
DAUPHIN_WIKI103_CONVS='[(850, 6)] * 3 + [(850, 1)] * 1 + [(850, 5)] * 4 + [(850, 1)] * 1 + [(850, 4)] * 3 + [(1024, 4)] * 1 + [(2048, 4)] * 1'
DAUPHIN_BWB_CONVS='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]'

function profile_meta() {
    python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-exact, max_vals = 1, val_interval = 10, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 0, d_hid = 128, batch_size = 4, val_data_limit = 100"
}

function profile_approx_meta() {
    python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-approx, max_vals = 1, val_interval = 10, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, d_hid = 128, batch_size = 4, val_data_limit = 100"
}

function meta_experiment() {
    echo "Not implemented"
}

function debug() {
    python -m ipdb main.py --config config/meta.conf --overrides "exp_name = metalearn-v2, run_name = debug-env, max_vals = 3, val_interval = 100, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2,mnli\", eval_tasks = \"none\", do_eval = 0, cuda = ${gpuid}, metatrain = 1, slow_params_approx = 0, sent_enc = convlm, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1, multistep_scale = 0.1"
}

# Wiki103 + MNLI + fine-tuning
function metapretrain() {
    python main.py --config config/meta.conf --overrides "train_tasks = \"mnli-alt,wiki103\", eval_tasks = \"mnli\", run_name = exact-mnli-wiki103-finetune-s${seed}, train_for_eval = 1, do_eval = 1, slow_params_approx = 0, one_sided_update = 1, batch_size = 8, mnli_classifier_hid_dim = 256, mnli-alt_pair_attn = 0, mnli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm , conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\", multistep_loss = 1, multistep_scale = 0.1" 
}

## No meta-learning ##
# MNLI and SNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = approx-mnli-snli-schdcos-r2, metatrain = 0, slow_params_approx = 0, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = 222, scheduler = cosine" 

# just Wiki103
#python main.py --config config/meta.conf --overrides "train_tasks = \"wiki103\", eval_tasks = \"wiki103\", run_name = nometa-wiki103-dauphin-s3, metatrain = 0, slow_params_approx = 0, batch_size = 32, lr = .00003, sim_lr = .00003, val_interval = 1000, max_vals = 1000, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 512, sent_enc = convlm , conv_layers = \"${DAUPHIN_WIKI103_CONVS}\""

# Wiki103 + MNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,wiki103\", eval_tasks = \"mnli,wiki103\", run_name = nometa-mnli-wiki103-lr.01-s2, metatrain = 0, slow_params_approx = 0, mnli_pair_attn = 0, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .01, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm, conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\""

# Wiki103 pretrain; MNLI finetune
#python main.py --config config/meta.conf --overrides "train_tasks = \"wiki103\", eval_tasks = \"mnli\", run_name = nometa-wiki103-mnli-finetune-s${seed}, train_for_eval = 1, do_eval = 1, metatrain = 0, slow_params_approx = 0, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm , conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\", multistep_loss = 1, multistep_scale = 0.01" 

if [ $1 == 'profile-meta' ]; then
    profile_meta
elif [ $1 == 'profile-approx-meta' ]; then
    profile_approx_meta
elif [ $1 == 'debug' ]; then
    debug
fi 
