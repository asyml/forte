# Copyright 2020 The Forte Authors. All Rights Reserved.
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
import argparse
import logging
import yaml

from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.base.data_selector_for_da import RandomDataSelector
from forte.data.selector import AllPackSelector
from forte.data.caster import MultiPackBoxer
from forte.processors.data_augment import ReplacementDataAugmentProcessor
from fortex.nltk import NLTKWordTokenizer, NLTKPOSTagger

logging.root.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="./config.yml", help="Config YAML filepath"
    )
    args = parser.parse_args()

    # loading config
    config = yaml.safe_load(open(args.config_file, "r"))

    nlp: Pipeline[MultiPack] = Pipeline()
    nlp.set_reader(RandomDataSelector(), config=config["data_selector_config"])
    nlp.add(component=MultiPackBoxer(), config=config["boxer_config"])
    nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
    nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())
    nlp.add(
        component=ReplacementDataAugmentProcessor(),
        config=config["da_processor_config"],
    )

    nlp.initialize()

    for _, m_pack in enumerate(nlp.process_dataset()):
        aug_pack = m_pack.get_pack("augmented_input")
        logging.info(aug_pack.text)


if __name__ == "__main__":
    main()
