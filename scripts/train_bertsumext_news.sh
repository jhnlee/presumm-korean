set -x

BERT_DATA_PATH='./datasets/bertabs_data_news/news'
MODEL_PATH='./checkpoint/bertsum_original/ext/news/'
LOG_PATH='./logs/bertsum/ext/ext_news50k.log'
python src/train.py  \
    -task ext \
    -mode train \
    -bert_data_path ${BERT_DATA_PATH} \
    -ext_dropout 0.1 \
    -model_path ${MODEL_PATH} \
    -lr 2e-3 \
    -visible_gpus 0,1 \
    -report_every 50 \
    -save_checkpoint_steps 1000 \
    -batch_size 3000 \
    -train_steps 50000 \
    -accum_count 3 \
    -use_interval true \
    -warmup_steps 10000 \
    -max_pos 512 \
    -log_file ${LOG_PATH}
