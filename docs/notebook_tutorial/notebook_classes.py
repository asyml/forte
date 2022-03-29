import os
import shutil
from typing import Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from forte import Pipeline
from forte.data import DataPack
from forte.common import Resources, Config
from forte.processors.base import PackProcessor
from forte.data.readers import PlainTextReader


class MachineTranslationProcessor(PackProcessor):
    """
    Translate the input text and output to a file.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # Initialize the tokenizer and model
        model_name: str = self.configs.pretrained_model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.task_prefix = "translate English to German: "
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.isdir(self.configs.output_folder):
            os.mkdir(self.configs.output_folder)

    def _process(self, input_pack: DataPack):
        file_name: str = os.path.join(
            self.configs.output_folder, os.path.basename(input_pack.pack_name)
        )

        # en2de machine translation
        inputs = self.tokenizer(
            [
                self.task_prefix + sentence
                for sentence in input_pack.text.split("\n")
            ],
            return_tensors="pt",
            padding=True,
        )

        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
        )

        outputs = self.tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )

        # Write output to the specified file
        with open(file=file_name, mode="w") as f:
            f.write("\n".join(outputs))

    @classmethod
    def default_configs(cls) -> Dict:
        return {
            "pretrained_model": "t5-small",
            "output_folder": "mt_test_output",
        }
