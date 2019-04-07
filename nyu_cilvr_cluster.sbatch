#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --constraint=gpu_12gb,pascal

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, pretrain_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.
<<<<<<< HEAD


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


function ultrafinebertlarge() {
    python main.py --config config/baselinesuperglue/superglue.conf --overrides " exp_name =ultrafine, run_name = bert_large-lr1e-5, pretrain_tasks = \"ultrafine\", target_tasks = \"ultrafine\", do_target_task_training = 0, do_pretrain = 1, do_full_eval = 1, write_preds=\"val\", max_seq_len = 128, bert_model_name=bert-base-uncased,  batch_size = 8, lr = .00001, max_epochs = 10"
}


ultrafinebertlarge
=======
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


## Winograd training with LR =0.01 with no encoder
python main.py --config_file config/benchmarkssuperglue/superglue.conf --overrides "sent_enc=\"null\", bert_model_name=bert-large-uncased, sep_embs_for_skip=1, pretrain_tasks =winograd-coreference, do_pretrain=1, project_dir=winograd, target_tasks=winograd-coreference, run_name = large_bert_cased_null_enclr0.001,  bert_embeddings_mode = top, lr = 0.001"
>>>>>>> winograd
