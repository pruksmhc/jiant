#!/bin/bash

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
    python main.py --config config/baselinesuperglue/superglue.conf --overrides " exp_name =ultrafine, run_name = bert_large-lr1e-5, pretrain_tasks = \"ultrafine\", target_tasks = \"none\", do_target_task_training = 0, do_full_eval = 1, write_preds=\"val\", max_seq_len = 128, bert_model_name=bert-base-uncased,  batch_size = 8, lr = .00001, max_epochs = 10"
}


ultrafinebertlarge