#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --constraint=gpu_12gb,pascal

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, pretrain_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.


# Quick-start: run this
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim"	
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	
module load anaconda3	
module load cuda 10.0	
source activate jiant_new	

eval_cmd="do_target_task_training = 1, do_full_eval = 1, batch_size = 8, write_preds = test, write_strict_glue_format = 1"

function gap_then_winograd() {
	python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, do_pretrain=1, do_target_task_training=1, target_tasks=winograd-coreference, run_name = null_enc_gap_then_winograd, project_dir=winograd, bert_embeddings_mode = top,  exp_name=bert_large_uncased, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased,"
}

gap_then_winograd