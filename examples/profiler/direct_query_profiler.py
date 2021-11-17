import os
import time
from typing import List, Tuple
import yaml
import texar.torch as tx
import codecs
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize


def preprocess(line):
    sentence: List[str] = []
    pos_tags: List[str] = []
    conll_components = line.split()


class OntonoteDataProcessing():
    def __init__(self, file_path = "data_samples/profiler/combine_data/test_combine.gold_conll"):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
            )
        )
        self.dataset_path = os.path.join(
            root_path, file_path
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        

    def data_processing(self):
        sentence = []
        pos_tags = []
        words = []
        tokenize_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        # get processed pack from dataset
        bg = time.time()
        
        with codecs.open(self.dataset_path, "r", encoding="utf8") as open_file:
            bg = time.time()
            sen = ''
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_components = line.strip().split()

                    sen += conll_components[3] + ' '
                    pos_tag = conll_components[4]

                    words.append(conll_components[3])
                    pos_tags.append(pos_tag)
                if line == "":
                    sentence.append(sen)
                    sen = ''
            process_time = time.time() - bg 

        # either use NLTK word_tokenize or self.tokenizer( Bert Tokenizer )
        t0 = time.time()
        for sent in sentence:
            # t1 = time.time()
            # tok = word_tokenize(sent)
            # tokenize_time += time.time() - t1
            pass
        get_sentence_time += time.time() - t0
 
        t2 = time.time()
        for w in words:
            t3 = time.time()
            tok = self.tokenizer(w, return_tensors="pt")
            tokenize_time += time.time() - t3
            # pass
        get_token_entity_time = time.time() - t2
        


        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_time": get_token_entity_time,
        }
        return time_dict


if __name__ == "__main__":
    data_pipeline = OntonoteDataProcessing()
    t1 = time.time()
    time_dict = data_pipeline.data_processing()
    t2 = time.time()
    print("total time spent: ", t2-t1)
    print(f"profiling time: {time_dict}")
