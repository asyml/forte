import os
import time
from typing import List, Tuple
import yaml
import texar.torch as tx
import codecs
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag



class OntonoteDataProcessing():
    def __init__(self, file_path = "data_samples/profiler/combine_data/development_combine.gold_conll"):
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
        document = []
        words = []
        subword_token = []
        NLTK_token = []
        NLTK_tags = []

        process_time = 0
        tokenize_time = 0
        get_sentence_time = 0
        get_doc_time = 0
        get_token_entity_time = 0
        # get processed pack from dataset
        bg = time.time()
        
        with codecs.open(self.dataset_path, "r", encoding="utf8") as open_file:
            sen = ''
            doc = ''
            for line in open_file:
                bg = time.time()
                line = line.strip()

                if line.startswith("#end document"):
                    document.append(doc)

                if line != "" and not line.startswith("#"):
                    conll_components = line.strip().split()
                    sen += conll_components[3] + ' '
                    words.append(conll_components[3])
                    process_time += time.time() - bg 

                    t3 = time.time()
                    subword_token.append(self.tokenizer(conll_components[3], return_tensors="pt"))
                    tokenize_time += time.time() - t3

                if line == "":
                    t4 = time.time()
                    doc += sen
                    sentence.append(sen)
                    process_time += time.time() - t4

                    t1 = time.time()
                    text = word_tokenize(sen)
                    NLTK_token.append(text)
                    NLTK_tags.append(pos_tag(text))
                    tokenize_time += time.time() - t1
                    sen = ''

        # query NLTK word_tokenize or self.tokenizer( subword Tokenizer )

        t0 = time.time()
        for sent in sentence:
            pass    
        get_sentence_time = time.time() - t0

        t5 = time.time()
        for d in document:
            pass    
        get_doc_time = time.time() - t5
 
        t2 = time.time()
        for s in NLTK_token:
            pass
        for t in NLTK_tags:
            pass
        for w in subword_token:  
            pass
        get_token_entity_time = time.time() - t2

        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_time": get_token_entity_time,
            "get_doc_time": get_doc_time
        }
        return time_dict


if __name__ == "__main__":
    data_pipeline = OntonoteDataProcessing()
    t1 = time.time()
    time_dict = data_pipeline.data_processing()
    t2 = time.time()
    print("total time spent: ", t2-t1)
    print(f"profiling time: {time_dict}")
