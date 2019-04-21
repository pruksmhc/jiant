#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:p40:1
#SBATCH --mem=20GB
#SBATCH --time=72:00:00

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, pretrain_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.


# Quick-start: run this
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/beegfs/yp913/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim"	
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	
source activate jiant


function gap_then_winograd() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks = gap-coreference, do_pretrain=1, do_target_task_training=1, reload_vocab=1, reload_tasks=1, target_tasks=winograd-coreference, run_name = null_enc_gap_then_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", tokenizer=bert-large-uncased, bert_model_name = bert-large-uncased"
}


function only_winograd_large() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=winograd_hope, bert_embeddings_mode = top,  exp_name=bert_large_uncased, tokenizer=bert-large-uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}
function only_winograd() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd_try_again, project_dir=winograd_hope, bert_embeddings_mode = top,  exp_name=bert_base_uncased_try, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-base-uncased"
}

function winograd_with_dev_test_switched() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased_dev_test_switched, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}


function only_gap() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "pretrain_tasks = gap-coreference, do_pretrain=1, do_target_task_training=0, tokenizer=bert-large-uncased, target_tasks=gap-coreference, run_name = null_enc_only_gap, project_dir=gap_again, bert_embeddings_mode = top,  exp_name=bert_large_uncased_try, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}



function only_winograd_get_results() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=0, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, load_model =1, load_eval_checkpoint=\"/beegfs/yp913/winograd/winograd/bert_large_uncased/null_enc_only_winograd/model_state_main_epoch_2.best_macro.th\",  sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

function only_winograd_with_gap_get_results() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=0, write_preds=val, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_gap_then_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, load_model =1, load_eval_checkpoint=\"/beegfs/yp913/winograd/winograd/bert_large_uncased/null_enc_gap_then_winograd/model_state_main_epoch_12.best_macro.th\",  sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

only_gap

