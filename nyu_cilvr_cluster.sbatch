#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:p40:1
#SBATCH --mem=30GB
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

eval_cmd="do_target_task_training = 1, do_full_eval = 1, batch_size = 8, write_preds = test, write_strict_glue_format = 1"

function ultrafinebalanced() {
   python main.py --config config/baselinesuperglue/params.conf
}

function ultrafinebalancedbertsmall() {
   python main.py --config config/baselinesuperglue/params.conf --overrides "exp_name=bert-base-uncased, bert_model_name=bert-base-uncased, tokenizer=bert-base-uncased "
}

function ultrafinesamedistribution() {
	python main.py --config config/baselinesuperglue/superglue.conf --overrides "project_dir=ultrafine_same_distribution, pretrain_tasks=ultrafine-same-distribution, target_tasks=ultrafine-same-distribution, do_pretrain=1, do_full_eval = 1, batch_size=4,  tokenizer = \"bert-large-uncased\", bert_model_name = \"bert-large-uncased\""
}
function ultrafineunbalancedtrain() {
	python main.py --config config/baselinesuperglue/superglue.conf --overrides "exp_name = bert_large_uncased, project_dir=ultrafine, run_name = unbalanced_ultrafine_test, pretrain_tasks = \"ultrafine\", target_tasks = \"ultrafine\", do_target_task_training = 0, do_full_eval = 1, do_pretrain = 1, batch_size = 4, lr = .00001, max_epochs = 10, tokenizer = \"bert-large-uncased\", bert_model_name = \"bert-large-uncased\""
}

function ultrafineunbalancedtrain0.001() {
	python main.py --config config/baselinesuperglue/superglue.conf --overrides "exp_name = bert_large_uncased, project_dir=ultrafine, run_name = unbalanced_ultrafine_test0.001, pretrain_tasks = \"ultrafine\", target_tasks = \"ultrafine\", do_target_task_training = 0, do_full_eval = 1, do_pretrain = 1, batch_size = 4, lr = .001, max_epochs = 10, tokenizer = \"bert-large-uncased\", bert_model_name = \"bert-large-uncased\""
}

function gap_then_winograd() {
    python main.py --config config/superglue.conf --overrides "exp_name = bert-large-cased, project_dir=winograd run_name = gap_then_winograd, pretrain_tasks = \"gap\", target_tasks = \"commitbank\", do_target_task_training = 1, do_full_eval = 1, max_seq_len = 128, batch_size = 4, lr = .00001, max_epochs = 10, tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\""
}

function get_results() {
    python main.py --config config/baselinesuperglue/superglue.conf --overrides "exp_name =bert_large_uncased, project_dir=ultrafine_balanced, write_preds=test, run_name =null_enc_fine_tune, pretrain_tasks = \"ultrafine-balanced\", target_tasks = \"ultrafine-balanced\", do_target_task_training = 0, do_full_eval = 1,  batch_size = 2, lr = .00001, max_epochs = 10, tokenizer = \"bert-large-uncased\", load_model = 1, bert_fine_tune=0, load_eval_checkpoint=\"/beegfs/yp913/jiant/ultrafine_balanced/bert_large_uncased_412_maxseqlen/null_enc_fine_tune/model_state_main_epoch_4.best_macro.th\",  bert_model_name = \"bert-large-uncased\""
}

get_results


