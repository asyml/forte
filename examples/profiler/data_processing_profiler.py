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
from forte.data.data_tuple import DataTuple
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from forte.processors.nlp import CoNLLNERPredictor
from ft.onto.base_ontology import Sentence, Document, Token, EntityMention
from fortex.nltk import NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter


class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataTuple):
        pass

class OntonoteDataProcessing():
    def __init__(self, file_path = "data_samples/profiler/combine_data"):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
            )
        )
        self.dataset_path = os.path.join(
            self.root_path, file_path
        )
        # Define and config the Pipeline
        self.nlp = Pipeline[DataTuple]()
        self.nlp.set_reader(OntonotesReader())
        self.nlp.add(DummyPackProcessor())
        self.nlp.add(NLTKSentenceSegmenter())
        self.nlp.add(NLTKWordTokenizer())
        self.nlp.add(NLTKPOSTagger())
        
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
        iter_time = 0
        get_token_entity_from_document_time = 0
        # get processed pack from dataset
        bg = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        # print("length of iterator: ", len(iter))
        process_time = time.time() - bg

        """
        iter_time:
        Iterate through the dataset and get each datapack
        """
        it = time.time()
        packcnt = 0
        sentcnt = 0
        token_sent_cnt = 0
        token_doc_cnt = 0
        for i, pack in enumerate(iter):
            packcnt +=1
            iter_time += time.time() - it
            
            """
                get_document_time:
                Get document from databack (expected to have only one document each pack)
            """
            dt = time.time()
            document = []
            for doc in pack.get_raw(Document):
                document.append(doc)
            if len(document) > 1:
                raise RuntimeError("More than one document in a datatuple")
            document = document[0]
            document_text = pack.get_text(document)
            get_document_time += time.time() - dt

            """
                get_sentence_time:
                Get sentence from databack
            """
            st = time.time()
            sentences = []
            for sent in pack.get_raw(Sentence):
                sentcnt += 1
                sent_text = pack.get_text(sent)
                sentences.append(sent_text)
                get_sentence_time += time.time()-st
                
                """
                get_token_entity_time: 
                Get token and entity from every sentence of the document
                """
                tet = time.time()
                tokenized_tokens = []
                tokens = [pack.get_text(t) for t in pack.get_raw(Token, sent)]
                token_sent_cnt += len(tokens)
                entities = [pack.get_text(entity)
                    for entity in pack.get_raw(EntityMention, sent)
                ]
                get_token_entity_time += time.time() - tet

                """
                tokenize_time: 
                Tokenize tokens and entities with BERTTokenizer
                """
                tt = time.time()
                tokenized_tokens += self.tokenizer.map_text_to_id(sent_text)
                token_ids = self.tokenizer.map_token_to_id(tokens)
                entity_ids = self.tokenizer.map_token_to_id(entities)
                tokenize_time += time.time()-tt
                
                st = time.time()
            
            """
            get_token_entity_from_document_time:
            Get token and entity from the whole document
            """
            tokdocTime = time.time()
            tokenized_tokens = []
            tokens = [pack.get_text(t) for t in pack.get_raw(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get_raw(EntityMention, document)
            ]
            get_token_entity_from_document_time += time.time() - tokdocTime

            """
            get_data_time:
            Get sentence attributes with get_data request
            """
            datat = time.time()
            sentences = []
            request = {
                Sentence: ["speaker", "part_id"]
            }
            for sent in pack.get_data_raw(Sentence, request):
                sentences.append(sent['context'])
            
            sentences_text_comp = " ".join(sentences)
            get_data_time += time.time() - datat
            
            it = time.time()
        
        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_from_sentence_time": get_token_entity_time,
            "get_token_entity_from_document_time": get_token_entity_from_document_time,
            "get_data_time": get_data_time,
            "iter_time": iter_time
        }
        print("datapack count:", packcnt)
        print("sentence count:", sentcnt)
        print("tokens count", token_sent_cnt, token_doc_cnt)
        return time_dict
    
    def whole_data_processing(self):
        file_path = "data_samples/profiler/whole_data"
        dataset_path = self.dataset_path = os.path.join(
            self.root_path, file_path
        )

        tokenize_time = 0
        get_document_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        get_data_time = 0
        iter_time = 0
        get_token_entity_from_document_time = 0
        # get processed pack from dataset
        bg = time.time()
        iter = self.nlp.process_dataset(dataset_path)
        # print("length of iterator: ", len(iter))
        process_time = time.time() - bg

        """
        iter_time:
        Iterate through the dataset and get each datapack
        """
        it = time.time()
        packcnt = 0
        sentcnt = 0
        token_sent_cnt = 0
        token_doc_cnt = 0
        for i, pack in enumerate(iter):
            packcnt +=1
            iter_time += time.time() - it
            
            """
                get_document_time:
                Get document from databack (expected to have only one document each pack)
            """
            dt = time.time()
            document = []
            for doc in pack.get_raw(Document):
                document.append(doc)
            if len(document) > 1:
                raise RuntimeError("More than one document in a datatuple")
            document = document[0]
            document_text = pack.get_text(document)
            get_document_time += time.time() - dt

            """
                get_sentence_time:
                Get sentence from databack
            """
            st = time.time()
            sentences = []
            for sent in pack.get_raw(Sentence):
                sentcnt += 1
                sent_text = pack.get_text(sent)
                sentences.append(sent_text)
                get_sentence_time += time.time()-st
                
                """
                get_token_entity_time:
                Get token and entity from every sentence of the document
                """
                tet = time.time()
                tokenized_tokens = []
                tokens = [pack.get_text(t) for t in pack.get_raw(Token, sent)]
                token_sent_cnt += len(tokens)
                entities = [pack.get_text(entity)
                    for entity in pack.get_raw(EntityMention, sent)
                ]
                get_token_entity_time += time.time() - tet

                """
                tokenize_time: 
                Tokenize tokens and entities with BERTTokenizer
                """
                tt = time.time()
                tokenized_tokens += self.tokenizer.map_text_to_id(sent_text)
                token_ids = self.tokenizer.map_token_to_id(tokens)
                entity_ids = self.tokenizer.map_token_to_id(entities)
                tokenize_time += time.time()-tt
                
                st = time.time()
            
            """
            get_token_entity_from_document_time:
            Get token and entity from the whole document
            """
            tokdocTime = time.time()
            tokenized_tokens = []
            tokens = [pack.get_text(t) for t in pack.get_raw(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get_raw(EntityMention, document)
            ]
            get_token_entity_from_document_time += time.time() - tokdocTime

            """
            get_data_time:
            Get sentence attributes with get_data request
            """
            datat = time.time()
            sentences = []
            request = {
                Sentence: ["speaker", "part_id"]
            }
            for sent in pack.get_data_raw(Sentence, request):
                sentences.append(sent['context'])
            
            sentences_text_comp = " ".join(sentences)
            get_data_time += time.time() - datat
            
            it = time.time()
        
        time_dict = {
            "process_time": process_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_from_sentence_time": get_token_entity_time,
            "get_token_entity_from_document_time": get_token_entity_from_document_time,
            "get_data_time": get_data_time,
            "iter_time": iter_time
        }
        print("datapack count:", packcnt)
        print("sentence count:", sentcnt)
        print("tokens count", token_sent_cnt, token_doc_cnt)
        return time_dict

    def deserialize_whole_datapack(self):
        tokenize_time = 0
        get_document_time = 0
        get_sentence_time = 0
        get_token_entity_time = 0
        get_data_time = 0
        iter_time = 0
        get_token_entity_from_document_time = 0
        deserialize_time = 0

        dese_t = time.time()
        output_file_path = "data_samples/profiler/combine_data/1pack.pkl"
        output_dataset_path = os.path.join(self.root_path, output_file_path)
        pack = DataTuple.deserialize(output_file_path, serialize_method = "pickle")
        deserialize_time = time.time() - dese_t
        
        """
            get_document_time:
            Get document from datapack
        """
        dt = time.time()
        documents = []
        for doc in pack.get_raw(Document):
            documents.append(doc)
            document_text = pack.get_text(doc)
        
        get_document_time += time.time() - dt

        """
            get_sentence_time:
            Get sentence from databack
        """
        st = time.time()
        sentences = []
        tokens_sent_cnt = 0
        for sent in pack.get_raw(Sentence):
            
            sent_text = pack.get_text(sent)
            sentences.append(sent_text)
            get_sentence_time += time.time()-st
            
            """
            get_token_entity_time: 
            Get token and entity from every sentence of the document
            """
            tet = time.time()
            tokens = [pack.get_text(t) for t in pack.get_raw(Token, sent)]
            tokens_sent_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get_raw(EntityMention, sent)
            ]
            get_token_entity_time += time.time() - tet

            """
            tokenize_time: 
            Tokenize tokens and entities with BERTTokenizer
            """
            tt = time.time()
            tokenized_tokens = self.tokenizer.map_text_to_id(sent_text)
            token_ids = self.tokenizer.map_token_to_id(tokens)
            entity_ids = self.tokenizer.map_token_to_id(entities)
            tokenize_time += time.time()-tt
            
            st = time.time()
        
        """
        get_token_entity_from_document_time:
        Get token and entity from the whole document
        """
        tokdocTime = time.time()
        tokens_doc_cnt = 0
        for document in pack.get_raw(Document):
            tokens = [pack.get_text(t) for t in pack.get_raw(Token, document)]
            tokens_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get_raw(EntityMention, document)
            ]
        get_token_entity_from_document_time += time.time() - tokdocTime

        """
        get_data_time:
        Get sentence attributes with get_data request
        """
        datat = time.time()
        sentences = []
        request = {
            Sentence: ["speaker", "part_id"]
        }
        for sent in pack.get_data_raw(Sentence, request):
            sentences.append(sent['context'])
        
        sentences_text_comp = " ".join(sentences)
        get_data_time += time.time() - datat
        
    
        time_dict = {
            "deserialize_time": deserialize_time,
            "tokenize_time": tokenize_time,
            "get_document_time": get_document_time,
            "get_sentence_time": get_sentence_time,
            "get_token_entity_from_sentence_time": get_token_entity_time,
            "get_token_entity_from_document_time": get_token_entity_from_document_time,
            "get_data_time": get_data_time,
            "iter_time": iter_time
        }
        print("docuemnt count:", len(documents))
        print("sentence count:", len(sentences))
        print("tokens count", tokens_sent_cnt, tokens_doc_cnt)
        return time_dict


if __name__ == "__main__":
    data_pipeline = OntonoteDataProcessing()
    t1 = time.time()
    time_dict = data_pipeline.data_processing()
    # time_dict = data_pipeline.whole_data_processing()
    # time_dict = data_pipeline.deserialize_whole_datapack()
    t2 = time.time()
    print("total time spent: ", t2-t1)
    print(f"profiling time: {time_dict}")
