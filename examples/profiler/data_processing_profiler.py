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

from collections import defaultdict
import os
import time
import random
from typing import List, Tuple
from ddt import data
from sortedcontainers.sortedlist import SortedList
import yaml
from termcolor import colored
import texar.torch as tx


from forte.common.configuration import Config
from forte.data.data_tuple import DataTuple
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from forte.processors.nlp import CoNLLNERPredictor
from forte.utils.types import T
from ft.onto.base_ontology import Sentence, Document, Token, EntityMention
from fortex.nltk import NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter


class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataTuple):
        pass

class TestObj:
    def __init__(self, id, begin, end, type):
        self.id = id
        self.begin = begin
        self.end = end
        self.type = type

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
    
    
    def data_constructor_profiling(self):
        # construct with dictionary
        test_text = "for text purpose"
        random_range = []
        for i in range(100000):
            random_range.append((random.randint(0, i), random.randint(i, i*2)))

        t0 = time.time()
        # objects = SortedList(key = lambda x: (x[1], x[2]))
        objects = []
        for i in range(100000):
            obj = [Sentence, random_range[i][0], random_range[i][1], i]
            obj.append(("text", test_text))
            objects.append(obj)
            # objects.add(obj)
        tuple_time = time.time() - t0

        t1 = time.time()
        # objects = SortedList(key = lambda x: (x["begin"], x["end"]))
        objects = []
        for i in range(100000):
            obj = {"type": Sentence, "begin": random_range[i][0], "end": random_range[i][1], "id": i}
            obj["text"] = test_text
            # objects.add(obj)
            objects.append(obj)
        dic_time = time.time() - t1
            
        t2 = time.time()
        # objects = SortedList(key = lambda x: (x.begin, x.end))
        objects = []
        for i in range(100000):
            obj = TestObj(i, random_range[i][0], random_range[i][1], Sentence)
            obj.text = test_text
            # objects.add(obj)
            objects.append(obj)
        obj_time = time.time() - t2

        t3 = time.time()
        sent_struct = {"id":0, "begin":1, "end":2, "text":3, "part_id":4, "sentiment":5, "classification": 6, "classifications": 7}
        # objects = SortedList(key = lambda x: (x[sent_struct["begin"]], x[sent_struct["end"]]))
        objects = []
        for i in range(100000):
            obj = [i, random_range[i][0], random_range[i][1], None, None, None, None, None]
            obj[sent_struct["text"]] = test_text
            # objects.add(obj)
            objects.append(obj)
        df_time = time.time() - t3

        time_dict = {
            "tuple_construct": tuple_time,
            "dictionary_construct": dic_time,
            "class_construct": obj_time,
            "tuple_dataframe_constrcut": df_time,
        }
        return time_dict

    def data_getter_profiling(self):
        # construct with dictionary
        test_text = "for text purpose"
        random_range = []
        for i in range(100000):
            random_range.append((random.randint(0, i), random.randint(i, i*2)))

        objects = SortedList(key = lambda x: (x[1], x[2]))
        for i in range(100000):
            # if i%3 == 0:
            if 0:
                obj = [Sentence, random_range[i][0], random_range[i][1], i]
            else:
                obj = [Token, random_range[i][0], random_range[i][1], i]
            obj.append(("text", test_text))
            objects.add(obj)

        def get_text(dic, key):
            for (k, v) in dic[4:]:
                if k == key:
                    return v
            return None

        t0 = time.time()
        sentences = []
        for obj in objects:
            if obj[0] == Token:
                sentences.append(get_text(obj, "text"))
        tuple_time = time.time() - t0
        
        objects = SortedList(key = lambda x: (x["begin"], x["end"]))
        for i in range(100000):
            # if i%3 == 0:
            if 0:
                obj = {"type": Sentence, "begin": random_range[i][0], "end": random_range[i][1], "id": i}
            else:
                obj = {"type": Token, "begin": random_range[i][0], "end": random_range[i][1], "id": i}
            obj["text"] = test_text
            objects.add(obj)

        t1 = time.time()
        sentences = []
        for obj in objects:
            if obj["type"] == Token:
                sentences.append(obj["text"])
        dic_time = time.time() - t1
            
        objects = SortedList(key = lambda x: (x.begin, x.end))
        for i in range(100000):
            # if i%3 == 0:
            if 0:
                obj = TestObj(i, random_range[i][0], random_range[i][1], Sentence)
            else:
                obj = TestObj(i, random_range[i][0], random_range[i][1], Token)
            obj.text = test_text
            objects.add(obj)

        t2 = time.time()
        sentences = []
        for obj in objects:
            if obj.type == Token:
                sentences.append(obj.text)
        obj_time = time.time() - t2

        sent_struct = {"id":0, "begin":1, "end":2, "text":3, "part_id":4, "sentiment":5, "classification": 6, "classifications": 7}
        tok_struct = {"id":0, "begin":1, "end":2, "text":3, "part_id":4, "sentiment":5, "classification": 6, "classifications": 7}
        sentence_objects = SortedList(key = lambda x: (x[sent_struct["begin"]], x[sent_struct["end"]]))
        token_objects = SortedList(key = lambda x: (x[tok_struct["begin"]], x[tok_struct["end"]]))
        for i in range(100000):
            # if i%3 == 0:
            if 0:
                obj = [i, random_range[i][0], random_range[i][1], None, None, None, None, None]
                obj[sent_struct["text"]] = test_text
                sentence_objects.add(obj)
            else:
                obj = [i, random_range[i][0], random_range[i][1], None, None, None, None, None]
                obj[tok_struct["text"]] = test_text
                token_objects.add(obj)

        t3 = time.time()
        sentences = []
        for obj in token_objects:
            sentences.append(obj[tok_struct["text"]])
        df_time = time.time() - t3

        time_dict = {
            "tuple_construct": tuple_time,
            "dictionary_construct": dic_time,
            "class_construct": obj_time, 
            "tuple_df_construct": df_time
        }
        return time_dict

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
            for doc in pack.get(Document):
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
            for sent in pack.get(Sentence):
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
                tokens = [pack.get_text(t) for t in pack.get(Token, sent)]
                token_sent_cnt += len(tokens)
                entities = [pack.get_text(entity)
                    for entity in pack.get(EntityMention, sent)
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
            tokens = [pack.get_text(t) for t in pack.get(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get(EntityMention, document)
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
            for sent in pack.get_data(Sentence, request):
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
            for doc in pack.get(Document):
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
            for sent in pack.get(Sentence):
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
                tokens = [pack.get_text(t) for t in pack.get(Token, sent)]
                token_sent_cnt += len(tokens)
                entities = [pack.get_text(entity)
                    for entity in pack.get(EntityMention, sent)
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
            tokens = [pack.get_text(t) for t in pack.get(Token, document)]
            token_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get(EntityMention, document)
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
            for sent in pack.get_data(Sentence, request):
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
    
    def iterator_profiling(self):
        file_path = "data_samples/profiler/whole_data"
        dataset_path = self.dataset_path = os.path.join(
            self.root_path, file_path
        )
        iter = self.nlp.process_dataset(dataset_path)
        co_it = time.time()
        for pack in iter:
            sentences = []
            tokens = []
            for ele in pack.co_iterate(Sentence, Token):
                if ele[0] == Sentence:
                    sentences.append(pack.get_text(ele))
                else:
                    tokens.append(pack.get_text(ele))
        print("co-iterate", len(sentences), len(tokens))
        co_iterate_time = time.time() - co_it

        iter = self.nlp.process_dataset(dataset_path)
        sep_it = time.time()
        for pack in iter:
            sentences = []
            tokens = []
            for sent in pack.get(Sentence):
                sentences.append(pack.get_text(sent))
                for tok in pack.get(Token, sent):
                    tokens.append(pack.get_text(tok))
        print("nested iterate", len(sentences), len(tokens))
        separate_iterate_time = time.time() - sep_it

        # iter = self.nlp.process_dataset(self.dataset_path)
        # part_it = time.time()
        # prev_sent = None
        # for pack in iter:
        #     sentences = []
        #     tokens = []
        #     for sent in pack.get(Sentence):
        #         sentences.append(pack.get_text(sent))
        #         if prev_sent is not None:
        #             for tok in pack.part_iterate(prev_sent, sent, Token):
        #                 tokens.append(pack.get_text(tok))
        #         prev_sent = sent
        # part_iterate_time = time.time() - part_it

        time_dict = {
            "co_iterate_time": co_iterate_time, 
            "separate_iterate_time": separate_iterate_time,
            # "part_iterate_time": part_iterate_time
        }
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
        for doc in pack.get(Document):
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
        for sent in pack.get(Sentence):
            
            sent_text = pack.get_text(sent)
            sentences.append(sent_text)
            get_sentence_time += time.time()-st
            
            """
            get_token_entity_time: 
            Get token and entity from every sentence of the document
            """
            tet = time.time()
            tokens = [pack.get_text(t) for t in pack.get(Token, sent)]
            tokens_sent_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get(EntityMention, sent)
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
        for document in pack.get(Document):
            tokens = [pack.get_text(t) for t in pack.get(Token, document)]
            tokens_doc_cnt += len(tokens)
            entities = [pack.get_text(entity)
                for entity in pack.get(EntityMention, document)
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
        for sent in pack.get_data(Sentence, request):
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
    ave_time = defaultdict(int)
    n = 5
    for i in range(n):
        time_dict = data_pipeline.data_constructor_profiling()
        # time_dict = data_pipeline.iterator_profiling()
        print(time_dict)
        for key, val in time_dict.items():
            ave_time[key] += val
    # time_dict = data_pipeline.iterator_profiling()
    # time_dict = data_pipeline.data_processing()
    # time_dict = data_pipeline.whole_data_processing()
    # time_dict = data_pipeline.deserialize_whole_datapack()
    t2 = time.time()
    for key, val in ave_time.items():
        ave_time[key] = val / n
    print("total time spent: ", t2-t1)
    print(f"average profiling time in {n} runs: {ave_time}")
