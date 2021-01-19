set -x

BERT_DATA_PATH='./datasets/bertext_data_news/news'
OUTPUT_PATH='./outputs/bertsumext/news/news'
CHECKPOINT_PATH='./checkpoint/bertsum_original/ext/news/model_step_24000.pt'
LOGDIR='./logs/bertsum/ext/news.log'

python src/bertsum/train.py \
    -task ext \
    -mode test \
    -batch_size 1000 \
    -test_batch_size 1000 \
    -bert_data_path ${BERT_DATA_PATH} \
    -log_file ${LOGDIR} \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_length 200 \
    -alpha 0.95 \
    -min_length 50 \
    -test_from ${CHECKPOINT_PATH} \
    -result_path ${OUTPUT_PATH} \
    -report_rouge false \
    -block_trigram false

exec > >(ts "%m/%d/%Y %H:%M:%S"| tee -a ${LOGDIR}) 2>&1 
python src/bertsum/cal_kor_rouge.py \
    --candidate_path './outputs/bertsumext/news/news_step24000.candidate' \
    --save_path './result/bertsumext/news/'
