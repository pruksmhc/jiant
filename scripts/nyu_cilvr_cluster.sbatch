#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
# If you're using a SLURM-managed cluster, modify this to suit your environment.

#SBATCH --gres=gpu:p40:1
#SBATCH --mem=20GB
#SBATCH --time=24:00:00

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



function gap_then_winograd_less_gap() {
 python main.py --config_file config/params_g_then_b.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = \"gap-coreference,winograd-coreference\", allow_reuse_of_pretraining_parameters = 1, do_pretrain=1, do_target_task_training=1, reload_vocab=1, reload_tasks=1, target_tasks=winograd-coreference, run_name = gap_then_winograd_proportional, project_dir=gap_then_winograd_final, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", tokenizer=bert-large-uncased, bert_model_name = bert-large-uncased"
}

function only_winograd_large() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = winograd-coreference, reload_tasks = 1, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=glue_4, bert_embeddings_mode = top,  exp_name=bert_large_uncased,  tokenizer=bert-large-uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

function only_winograd_bert_case() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = winograd-coreference, reload_tasks = 1, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=glue_3, bert_embeddings_mode = top,  exp_name=bert_large_cased,  tokenizer=bert-large-cased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"none\", bert_model_name = bert-large-cased"
}

function only_winograd_bert_case_bert_adam() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, optimizer=bert_adam, pretrain_tasks = winograd-coreference, reload_tasks = 1, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd_adam, project_dir=glue_3, bert_embeddings_mode = top,  exp_name=bert_large_cased,  tokenizer=bert-large-cased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"none\", bert_model_name = bert-large-cased"
}


function only_winograd_large_test() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = winograd-coreference, reload_tasks = 1, do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd, project_dir=winograd_try_selfattention_again_test, bert_embeddings_mode = top,  exp_name=bert_large_uncased,  tokenizer=bert-large-uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

function only_winograd_cbow() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "pretrain_tasks = winograd-coreference, bert_model_name=\"\", do_pretrain=1, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_only_winograd_try_again, project_dir=glue_cbow, bert_embeddings_mode = top,  exp_name=cbow, batch_size=16, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"none\", word_embs=\"glove\", skip_embs=0, sent_enc=bow, tokenizer=\"MosesTokenizer\", bert_fine_tune=0"
}

function only_gap() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = gap-coreference, do_pretrain=1, do_target_task_training=0, tokenizer=bert-large-uncased, target_tasks=gap-coreference, run_name = null_enc_only_gap, project_dir=gap_again_seflattention , bert_embeddings_mode = top,  exp_name=bert_large_uncased_try, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

function only_winograd_get_results() {
 python main.py --config_file config/params_g_then_b.conf --overrides "pretrain_tasks = winograd-coreference, write_preds=\"val,test\", do_pretrain=0, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_gap_then_winograd, project_dir=winograd_new_gap, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, load_model =1, load_eval_checkpoint=\"/beegfs/yp913/jiant/winograd_new_gap/bert_large_uncased/null_enc_gap_then_winograd/model_state_main_epoch_4.best_macro.th\",  sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

function only_winograd_with_gap_get_results() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks = winograd-coreference, do_pretrain=0, write_preds=val, do_target_task_training=0, target_tasks=winograd-coreference, run_name = null_enc_gap_then_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, load_model =1, load_eval_checkpoint=\"/beegfs/yp913/winograd/winograd/bert_large_uncased/null_enc_gap_then_winograd/model_state_main_epoch_12.best_macro.th\",  sent_enc = \"null\", bert_model_name = bert-large-uncased"
}

only_winograd_bert_case
