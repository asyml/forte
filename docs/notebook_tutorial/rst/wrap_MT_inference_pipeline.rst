Machine Translation Inference Pipeline
======================================

Packages
--------

.. code:: ipython3

    import os
    import shutil
    from typing import Dict
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from forte import Pipeline
    from forte.data import DataPack
    from forte.common import Resources, Config
    from forte.processors.base import PackProcessor
    from forte.data.readers import PlainTextReader


Background
----------

After a DS is satisfied with the results of a training model, they will
had their notebook over to an MLE who has to convert their model into an
inference model.

Inference Workflow
------------------

Pipeline
~~~~~~~~

To simplify the example, we consider ``t5-small`` as a trained MT model.
We should always consider pipeline first when it comes to an inference
workflow. As the
`glossary <https://asyml-forte.readthedocs.io/en/latest/index_appendices.html#glossary>`__
suggests, it’s an inference system that contains a set of processing
components.

Therefore, we initialize a ``pipeline`` below.

.. code:: ipython3

    pipeline: Pipeline = Pipeline[DataPack]()

Reader
~~~~~~

After observing the dataset, it’s a plain ``txt`` file. Therefore, we
can use ``PlainTextReader`` directly.

.. code:: ipython3

    pipeline.set_reader(PlainTextReader())

However, it’s still beneficial to take a deeper look at how this class
is designed so that users can customize a reader when needed.

Processor
~~~~~~~~~

We already have inference model which is the ``t5-small``, and we need a
component to make inference. Therefore, besides model itself, there are
several behaviors needed. 1. tokenization that transforms input text to
sequences of tokens. 2. since T5 has a better performance given a task
prompt, we also want to include the prompt in our data.

In forte, we have a generic class ``PackProcessor`` that wraps model and
inference related components and behaviors to process ``DataPack``, and
we need to create a class inherit generic method and customize the
behaviors.

The generic method to process ``DataPack`` is
``_process(self, input_pack: DataPack)``. It should tokenize the input
text, use class model to make an inference, decode the output token ids
and finally writes the output to a target file.

Given what we discussed, we have a processor class below, and we need to
add it to pipeline after define it.

.. code:: ipython3


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
            inputs = self.tokenizer([
                self.task_prefix + sentence
                for sentence in input_pack.text.split('\n')
            ], return_tensors="pt", padding=True)

            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
            )

            outputs = self.tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )

            # Write output to the specified file
            with open(file=file_name, mode='w') as f:
                f.write('\n'.join(outputs))

        @classmethod
        def default_configs(cls) -> Dict:
            return {
                "pretrained_model": "t5-small",
                "output_folder": "mt_test_output"
            }

    pipeline.add(MachineTranslationProcessor(), config={
        "pretrained_model": "t5-small"
    })

Examples
--------

We have an working `MT translation pipeline
example <https://github.com/asyml/forte/blob/master/docs/notebook_tutorial/wrap_MT_inference_pipeline.ipynb>`__.

There are several basic functions of processor and internal functions
are defined in this example.

-  ``initialize()``: Pipeline will call it at the start of processing.
   The processor will be initialized with ``configs``, and register
   global resources into ``resource``. The implementation should set up
   the states of the component.

   -  initialize a pre-trained model
   -  initialize tokenizer
   -  initialize model-specific attributes such as task prefix

-  ``process()``: using the loaded model to make predictions and write
   the prediction results out.

   -  we first tokenize the input text
   -  then use model to generate output sequence ids
   -  then we decode output sequence ids into tokens and write the
      output into a file

After setting up pipeline’s components, we can run the pipeline on the
input directory as below.

.. code:: ipython3

    dir_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(""))), "data_samples/machine_translation"
        )

    pipeline.run(dir_path)
    print("Done successfully")

One can investigate the machine translation output in folder
``mt_test_output`` located at the script location. Then we remove the
output folder below.

.. code:: ipython3

    shutil.rmtree(MachineTranslationProcessor.default_configs()["output_folder"])
