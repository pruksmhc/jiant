
#!/bin/bash

# Quick-start: run this
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim"	
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	
module load anaconda3	
module load cuda 10.0	
source activate jiant_new	

eval_cmd="do_target_task_training = 0, do_full_eval = 1, batch_size = 128, write_preds = test, write_strict_glue_format = 1"


## GAP training ##
python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 1, do_target_task_training = 0, lr=0.0001, run_name = small_bert_cased_rnn_enclr0.0001, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"rnn\""
python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 0, do_target_task_training = 1, lr=0.0001 run_name = small_bert_cased_rnn_enclr0.0001, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"rnn\", ${eval_cmd}"


python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 1, do_target_task_training = 0, lr=0.0001, run_name = small_bert_cased_no_enclr0.0001, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"null\""
python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 0, do_target_task_training = 1, lr=0.0001 run_name = small_bert_cased_no_enclr0.0001, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"null\", ${eval_cmd}"

python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 1, do_target_task_training = 0, lr=0.01, run_name = small_bert_cased_rnn_enclr0.01, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"rnn\""
python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 0, do_target_task_training = 1, lr=0.01 run_name = small_bert_cased_rnn_enclr0.01, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"rnn\", ${eval_cmd}"


python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 1, do_target_task_training = 0, lr=0.01, run_name = small_bert_cased_no_enclr0.01, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"null\""
python main.py --config_file config/benchmarkssuperglue/superglue.conf  --overrides "pretrain_tasks = ultrafine, target_tasks=ultrafine, do_pretrain = 0, do_target_task_training = 1, lr=0.01 run_name = small_bert_cased_no_enclr0.01, project_dir=ultrafine, bert_embeddings_mode = top, sent_enc = \"null\", ${eval_cmd}"
