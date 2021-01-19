set -x

BERT_DATA_PATH='./datasets/bertabs_data_news/news'
OUTPUT_PATH='./outputs/bertsumabs/news/news'
CHECKPOINT_PATH='./checkpoint/bertsum_original/abs/news/model_step_130000.pt'
LOGDIR='./logs/bertsum/abs/news.log'

python src/bertsum/train.py \
    -task abs \
    -mode test \
    -batch_size 3000 \
    -test_batch_size 3000 \
    -bert_data_path ${BERT_DATA_PATH} \
    -log_file ${LOGDIR} \
    -sep_optim true \
    -use_interval true \
    -visible_gpus 0 \
    -max_pos 512 \
    -max_tgt_len 300 \
    -alpha 0.95 \
    -min_length 50 \
    -report_rouge false \
    -result_path ${OUTPUT_PATH} \
    -test_from ${CHECKPOINT_PATH}

exec > >(ts "%m/%d/%Y %H:%M:%S"| tee -a ${LOGDIR}) 2>&1 
python src/bertsum/cal_kor_rouge.py \
    --candidate_path './outputs/bertsumabs/news/news.130000.candidate' \
    --save_path './result/bertsumabs/news/'
