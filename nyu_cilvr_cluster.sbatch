#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:p40:1
#SBATCH --mem=20GB
#SBATCH --time=72:00:00
# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, pretrain_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/beegfs/yp913/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim"	
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	
source activate jiant


function only_atomic() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks=atomic, do_pretrain=1, do_target_task_training=0, reload_vocab=1, reload_tasks=1, target_tasks=atomic, run_name = rnn_enc, project_dir=atomic, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"rnn\", tokenizer=bert-large-uncased, bert_model_name = bert-large-uncased"
}

function only_ontonotes() {
 python main.py --config_file config/baselinesuperglue/params.conf --overrides "pretrain_tasks=edges-coref-ontonotes, do_pretrain=1, do_target_task_training=0, reload_vocab=1, reload_tasks=1, target_tasks=edges-coref-ontonotes, run_name = rnn_enc, project_dir=ontonotes, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"rnn\", tokenizer=bert-large-uncased, bert_model_name = bert-large-uncased"
}


only_ontonotes