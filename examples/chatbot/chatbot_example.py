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
import yaml
from termcolor import colored
import torch
from fortex.nltk import NLTKSentenceSegmenter, NLTKWordTokenizer, NLTKPOSTagger
from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackTerminalReader
from forte.common.resources import Resources
from forte.pipeline import Pipeline
from forte.processors.third_party import MicrosoftBingTranslator
from forte.processors.nlp import SRLPredictor
from forte.processors.ir import SearchProcessor, BertBasedQueryCreator
from forte.data.selector import NameMatchSelector
from ft.onto.base_ontology import PredicateLink, Sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(config: Config) -> Pipeline:
    resource = Resources()
    query_pipeline = Pipeline[MultiPack](resource=resource)
    query_pipeline.set_reader(
        reader=MultiPackTerminalReader(), config=config.reader
    )
    query_pipeline.add(
        component=MicrosoftBingTranslator(), config=config.translator
    )
    query_pipeline.add(
        component=BertBasedQueryCreator(), config=config.query_creator
    )
    query_pipeline.add(component=SearchProcessor(), config=config.searcher)

    top_response_pack_name = config.indexer.response_pack_name_prefix + "_0"

    query_pipeline.add(
        component=NLTKSentenceSegmenter(),
        selector=NameMatchSelector(select_name=top_response_pack_name),
    )
    query_pipeline.add(
        component=NLTKWordTokenizer(),
        selector=NameMatchSelector(select_name=top_response_pack_name),
    )
    query_pipeline.add(
        component=NLTKPOSTagger(),
        selector=NameMatchSelector(select_name=top_response_pack_name),
    )
    query_pipeline.add(
        component=SRLPredictor(),
        config=config.SRL,
        selector=NameMatchSelector(select_name=top_response_pack_name),
    )
    query_pipeline.add(
        component=MicrosoftBingTranslator(), config=config.back_translator
    )

    query_pipeline.initialize()

    return query_pipeline


def main(config: Config):
    query_pipeline = setup(config)
    resource = query_pipeline.resource

    m_pack: MultiPack
    for m_pack in query_pipeline.process_dataset():
        # update resource to be used in the next conversation
        query_pack = m_pack.get_pack(config.translator.in_pack_name)
        if resource.get("user_utterance"):
            resource.get("user_utterance").append(query_pack)
        else:
            resource.update(user_utterance=[query_pack])

        response_pack = m_pack.get_pack(config.back_translator.in_pack_name)

        if resource.get("bot_utterance"):
            resource.get("bot_utterance").append(response_pack)
        else:
            resource.update(bot_utterance=[response_pack])

        english_pack = m_pack.get_pack("pack")
        print(
            colored("English Translation of the query: ", "green"),
            english_pack.text,
            "\n",
        )

        # Just take the first pack.
        pack = m_pack.get_pack(config.indexer.response_pack_name_prefix + "_0")
        print(colored("Retrieved Document", "green"), pack.text, "\n")
        print(
            colored("German Translation", "green"),
            m_pack.get_pack("response").text,
            "\n",
        )
        for sentence in pack.get(Sentence):
            sent_text = sentence.text
            print(colored("Sentence:", "red"), sent_text, "\n")

            print(colored("Semantic role labels:", "red"))
            for link in pack.get(PredicateLink, sentence):
                parent = link.get_parent()
                child = link.get_child()
                print(
                    f'  - "{child.text}" is role '
                    f"{link.arg_type} of "
                    f'predicate "{parent.text}"'
                )
            print()

            input(colored("Press ENTER to continue...\n", "green"))

if __name__ == "__main__":
    all_config = Config(yaml.safe_load(open("config.yml", "r")), None)
    main(all_config)
