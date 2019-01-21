#!/bin/bash

source user_config.sh
seed=222
gpuid=4

SMALLLM_CONVS='[(512, 3)] * 20'
BASELM_CONVS='[(1268, 4)] * 13'
MINIDAUPHIN_WIKI103_CONVS='[(850, 6)] * 3 + [(850, 1)] * 1 + [(850, 5)] * 4 + [(850, 1)] * 1 + [(850, 4)] * 3 + [(1024, 4)] * 1 + [(1024, 4)] * 1'
DAUPHIN_WIKI103_CONVS='[(850, 6)] * 3 + [(850, 1)] * 1 + [(850, 5)] * 4 + [(850, 1)] * 1 + [(850, 4)] * 3 + [(1024, 4)] * 1 + [(2048, 4)] * 1'
DAUPHIN_BWB_CONVS='[(512, 5)] + [(128, 1, 0), (128, 5, 0), (512, 1, 3)] * 3 + [(512, 1, 0), (512, 5, 0), (1024, 1, 3)] * 3 + [(1024, 1, 0), (1024, 5, 0), (2048, 1, 3)] * 6 + [(1024, 1, 0), (1024, 5, 0), (4096, 1, 3)]'

### debug ###
python -m ipdb main.py --config config/meta.conf --overrides "exp_name = metalearn, run_name = debug-meta, max_vals = 3, val_interval = 100, do_train = 1, train_for_eval = 0, train_tasks = \"mnli,snli\", eval_tasks = \"none\", do_eval = 0, cuda = ${gpuid}, metatrain = 0, slow_params_approx = 0, sent_enc = conv, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1, one_sided_update = 0"

# ConvLM 
#python -m ipdb main.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = debug-convlm, max_vals = 3, val_interval = 100, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2,mnli\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 0, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"
#python -m ipdb main.py --config config/meta.conf --overrides "train_tasks = \"wiki103\", eval_tasks = \"wiki103\", run_name = debug-convlm-params, metatrain = 0, slow_params_approx = 0, sent_enc = convlm, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, random_seed = ${seed}, scheduler = cosine, multistep_loss = 1, multistep_scale = 0.01, conv_layers = \"${DAUPHIN_WIKI103_CONVS}\"" 

# Non-meta, LM tasks
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = debug-convlm-wiki2, val_interval = 1000, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 0, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = debug-convlm-wiki103, val_interval = 1000, do_train = 1, train_for_eval = 0, train_tasks = \"wiki103\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 0, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"

# Meta, LM tasks
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = debug-meta-convlm, val_interval = 1000, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2,mnli\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 1, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"

# MNLI and SNLI
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, random_seed = ${seed}, run_name = debug-mnli-snli, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, approx_term = cos_sim"

# MNLI and MNLI-alt
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, random_seed = ${seed}, run_name = debug-mnli-mnli, train_tasks = \"mnli,mnli-alt\", metatrain = 1, slow_params_approx = 1, approx_term = cos_sim"

# Dot product
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, random_seed = ${seed}, run_name = debug-dot-prod-v2, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, approx_term = dot_product, lr = .00003, sim_lr = .00003, scheduler = cosine, max_sim_grad_norm = 100.0"

## profile ##
#python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-exact-nograph, max_vals = 1, val_interval = 10, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 0, d_hid = 128, batch_size = 4, val_data_limit = 100"
#python main_profile.py --config config/debug.conf --overrides "exp_name = metalearn, run_name = prof-approx, max_vals = 1, val_interval = 10, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, d_hid = 256, val_data_limit = 100"
#python main_profile.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = profile-convlm-exact, max_vals = 1, val_interval = 10, val_data_limit = 100, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2,mnli\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 1, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"
#python main_profile.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn, run_name = profile-convlm-approx, max_vals = 1, val_interval = 10, val_data_limit = 100, do_train = 1, train_for_eval = 0, train_tasks = \"wiki2,mnli\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 1, slow_params_approx = 1, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"


## Mess w/ approx method ##
seed=222
gpuid=5

# High cosine regularization weight
#python main.py --config config/meta.conf --overrides "cuda = ${gpuid}, random_seed = ${seed}, run_name = cossim-weight10-s2, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, approx_term = cos_sim, lr = .00003, sim_lr = 10.0, scheduler = cosine"
#python main.py --config config/meta.conf --overrides "cuda = ${gpuid}, random_seed = ${seed}, run_name = cossim-weight100-s2, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, approx_term = cos_sim, lr = .00003, sim_lr = 100.0, scheduler = cosine"
#python main.py --config config/meta.conf --overrides "cuda = ${gpuid}, random_seed = ${seed}, run_name = cossim-weight1000-s2, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, approx_term = cos_sim, lr = .00003, sim_lr = 1000.0, scheduler = cosine"

# Optimize cosine similarity
#python main.py --config config/meta.conf --overrides "cuda = ${gpuid}, random_seed = ${seed}, run_name = optimize-cossim-s2, train_tasks = \"mnli,snli\", metatrain = 1, slow_params_approx = 1, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, scheduler = cosine, approx_term = only_cos_sim"



### FEEL THE METALEARN ###
seed=222
gpuid=2

## Preprocess ##
#python main.py --config config/meta.conf --overrides "cuda = ${cuda}, exp_name = metalearn-lm, run_name = preproc-lm, val_interval = 1000, do_train = 1, train_for_eval = 0, train_tasks = \"bwb,wiki103\", eval_tasks = \"none\", do_eval = 0, cuda = ${cuda}, metatrain = 0, slow_params_approx = 0, sent_enc = convlm, skip_embs = 0, d_hid = 512, mnli_pair_attn = 0, snli_pair_attn = 0, multistep_loss = 1"

## Exact ##
# SNLI + MNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = exact-mnli-snli-schdcos-msl0.01-r0, slow_params_approx = 0, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, random_seed = ${seed}, scheduler = cosine, multistep_loss = 1, multistep_scale = 0.01" 

# Wiki103 + MNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,wiki103\", eval_tasks = \"mnli,wiki103\", run_name = exact-mnli-wiki103-s2, slow_params_approx = 0, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm , conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\", multistep_loss = 1, multistep_scale = 0.01" 

# Wiki103 + MNLI + fine-tuning
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,wiki103\", eval_tasks = \"mnli\", run_name = exact-mnli-wiki103-s2, train_for_eval = 1, slow_params_approx = 0, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm , conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\", multistep_loss = 1, multistep_scale = 0.01" 

## Approx ##
# MNLI and SNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = approx-mnli-snli-schdcos-r2, slow_params_approx = 1, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = 222, scheduler = cosine" 

# Wiki103 + MNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,wiki103\", eval_tasks = \"mnli,wiki103\", run_name = approx-mnli-wiki103-s2, slow_params_approx = 1, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm , conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\""


## No meta-learning ##
# MNLI and SNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,snli\", eval_tasks = \"mnli,snli\", run_name = approx-mnli-snli-schdcos-r2, metatrain = 0, slow_params_approx = 0, mnli_pair_attn = 0, snli_pair_attn = 0, lr = .00003, sim_lr = .00003, cuda = ${gpuid}, random_seed = 222, scheduler = cosine" 

# just Wiki103
#python main.py --config config/meta.conf --overrides "train_tasks = \"wiki103\", eval_tasks = \"wiki103\", run_name = nometa-wiki103-dauphin-s3, metatrain = 0, slow_params_approx = 0, batch_size = 32, lr = .00003, sim_lr = .00003, val_interval = 1000, max_vals = 1000, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 512, sent_enc = convlm , conv_layers = \"${DAUPHIN_WIKI103_CONVS}\""

# Wiki103 + MNLI
#python main.py --config config/meta.conf --overrides "train_tasks = \"mnli,wiki103\", eval_tasks = \"mnli,wiki103\", run_name = nometa-mnli-wiki103-lr.01-s2, metatrain = 0, slow_params_approx = 0, mnli_pair_attn = 0, batch_size = 8, mnli_classifier_hid_dim = 256, mnli_pair_attn = 0, lr = .01, cuda = ${gpuid}, random_seed = ${seed}, d_hid = 256, sent_enc = convlm, conv_layers = \"${MINIDAUPHIN_WIKI103_CONVS}\""
