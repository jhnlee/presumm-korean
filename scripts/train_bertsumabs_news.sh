set -x

BERT_DATA_PATH='./datasets/bertabs_data_news/news'
MODEL_PATH='./checkpoint/bertsum_original/abs/news/'
LOG_PATH='./logs/bertsum/abs/abs_news200k.log'
python src/train.py  \
    -task abs \
    -mode train \
    -bert_data_path ${BERT_DATA_PATH} \
    -dec_dropout 0.2  \
    -model_path ${MODEL_PATH} \
    -sep_optim true \
    -lr_bert 0.002 \
    -lr_dec 0.2 \
    -save_checkpoint_steps 5000 \
    -batch_size 140 \
    -train_steps 200000 \
    -report_every 50 \
    -accum_count 10 \
    -use_bert_emb true \
    -use_interval true \
    -warmup_steps_bert 20000 \
    -warmup_steps_dec 10000 \
    -max_pos 512 \
    -visible_gpus 0,1 \
    -log_file ${LOG_PATH}
