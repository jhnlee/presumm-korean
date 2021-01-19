sample_raw_path="./datasets/raw_data/news_sample"
sample_json_path="./datasets/json_data/news_sample"
sample_save_path="./datasets/bert_data/news_sample"
python preprocess.py -raw_path ${sample_raw_path} -json_path ${sample_json_path} -save_path ${sample_save_path}