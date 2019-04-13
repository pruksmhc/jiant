#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:p40:1
#SBATCH --mem=15GB
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

function ultrafineresult() {
    python main.py --config config/baselinesuperglue/superglue.conf --overrides "project_dir=ultrafine_exp,  exp_name = base_large_uncased, run_name = rnn_enc,  pretrain_tasks = \"ultrafine\", target_tasks = \"ultrafine\", do_target_task_training = 0, do_full_eval = 1, do_pretrain=0, load_model = 1, load_eval_checkpoint=\"/beegfs/yp913/jiant/ultrafine_exp/base_large_uncased/rnn_enc/model_state_main_epoch_1.best_macro.th\", max_seq_len = 510, batch_size = 4, lr = .00001, max_epochs = 10, tokenizer = \"bert-large-uncased\", bert_model_name = \"bert-large-uncased\""
}

function gap_then_winograd() {
    python main.py --config config/final-bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, exp_name = bert-large-cased, run_name = mnli_then_commitbank, pretrain_tasks = \"mnli\", target_tasks = \"commitbank\", do_target_task_training = 1, do_full_eval = 1, max_seq_len = 128, batch_size = 4, lr = .00001, max_epochs = 10, tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\""
}


## GAP training with null sent encoder,  lr = 0.0001 ## BERT LARGE
# this has the current 
python main.py --config config/baselinesuperglue/params.conf