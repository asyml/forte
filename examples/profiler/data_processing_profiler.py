# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import List, Tuple
import yaml
from termcolor import colored
import texar.torch as tx


from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from forte.processors.nlp import CoNLLNERPredictor
from ft.onto.base_ontology import Sentence, Document, Token, EntityMention
# from forte.nltk import NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter


class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        pass

class OntonoteDataProcessing():
    def __init__(self, file_path = "data_samples/profiler/combine_data"):
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
        # Define and config the Pipeline
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(OntonotesReader())
        self.nlp.add(DummyPackProcessor())
        # self.nlp.add(NLTKSentenceSegmenter())
        # self.nlp.add(NLTKWordTokenizer())
        # self.nlp.add(NLTKPOSTagger())
        
        # config = yaml.safe_load(open("config.yml", "r"))    
        # config = Config(config, default_hparams=None)
        # self.nlp.add(CoNLLNERPredictor(), config=config.NER)

        self.nlp.set_profiling(True)

        self.nlp.initialize()

        self.tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
        

    def data_processing(self):
        tokenize_time = 0
        get_document_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        get_data_time = 0
        # get processed pack from dataset
        bg = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        # print("length of iterator: ", len(iter))
        process_time = time.time() - bg

        for i, pack in enumerate(iter):
            """
            pack: DataTuple object
            """
            # print(f"DataTuple {i}, has {len(pack.elements)} elements")
            # get sentence from pack
            dt = time.time()
            document = []
            for doc in pack.get_raw(Document):
                document.append(doc)
            if len(document) > 1:
                raise RuntimeError("More than one document in a datatuple")
            document = document[0]
            document_text = pack.text(document)
            get_document_time += time.time() - dt

            """
                test get_raw and get_data_raw: 
                    get text of all sentences
                    get tokens from each sentence
                    compare the results of get_raw and get_data_raw
            """
            st = time.time()
            sentences = []
            for sent in pack.get_raw(Sentence):
                
                sent_text = pack.text(sent)
                sentences.append(sent_text)
                
                tet = time.time()
                tokenized_tokens = []
                tokens = [pack.get_raw(Token, sent)]
                entities = [pack.text(entity)
                    for entity in pack.get_raw(EntityMention, sent)
                ]
                get_token_entity_time += time.time() - tet

                tt = time.time()
                tokenized_tokens += self.tokenizer.map_text_to_id(sent_text)
                token_ids = self.tokenizer.map_token_to_id(tokens)
                entity_ids = self.tokenizer.map_token_to_id(entities)
                tokenize_time += time.time()-tt

                # print(colored("map_text_to_id:", "red"), len(tokenized_tokens), "\n")
                # print(colored("map_token_to_id:", "yellow"), len(token_ids), "\n")
                # print(colored("map_entity_to_id:", "yellow"), len(entity_ids), "\n")
            
            sentences_text = " ".join(sentences)
            assert(sentences_text == document_text)
            get_sentence_time += time.time()-st

            datat = time.time()
            sentences = []
            request = {
                Sentence: ["speaker", "part_id"]
            }
            for sent in pack.get_data_raw(Sentence, request):
                sentences.append(sent['context'])
            
            sentences_text_comp = " ".join(sentences)
            assert(sentences_text == sentences_text_comp)
            get_data_time += time.time() - datat
        
        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_time": get_token_entity_time,
            "get_data_time": get_data_time,
        }
        return time_dict

if __name__ == "__main__":
    data_pipeline = OntonoteDataProcessing()
    t1 = time.time()
    time_dict = data_pipeline.data_processing()
    t2 = time.time()
    print("total time spent: ", t2-t1)
    print(f"profiling time: {time_dict}")
