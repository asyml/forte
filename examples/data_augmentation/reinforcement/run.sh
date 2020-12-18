# under examples/data_augmentation/reinforcement dir
# prepare data
python prepare_data/download_imdb.py
python prepare_data/imdb_format.py --raw_data_dir=data/IMDB_raw/aclImdb --train_id_path=data/IMDB_raw/train_id_list.txt --output_dir=data/IMDB
python prepare_data/sample_train_validation_data_csv.py
python prepare_data/convert_bert_pkl.py
# run da_rl model
python main.py --do-train --do-eval --do-test
