Processor
==========

A pipeline component that wraps inference model and set up inference related work.



Examples
---------

We have an working MT translation pipeline example here https://github.com/asyml/forte/blob/master/docs/notebook_tutorial/wrap_MT_inference_pipeline.ipynb

There are several basic functions of processor and internal functions are defined in this example.

* `initialize()`: initilize model and MT related components/attributes
    - initialize a pre-trained model
    - intialize tokenizer
    - initialize model-specific attributes such as task prefix
* `process()`: using the loaded model to make predictions and write the prediction results out.
    - we first tokenize the input text
    - then use model to generate output sequence ids
    - then we decode output sequence ids into tokens and write the output into a file



Usage
------





Processor Class Hierarchy
------------------------------

Here we provide a simplified class hierarchy for :class:`MachineTranslationProcessor` to show the relations of processors which are subclasses of `PipelineComponent`.

* :class:`PipelineComponent`
    - :class:`BasePackProcessor`: The base class of processors that process one pack in a streaming way
        * :class:`PackProcessor`: The base class of processors that process one :class:`DataPack` each
    time.
            - :class:`MachineTranslationProcessor`
        * :class:`MultiPackProcessor`: The base class of processors that process :class:`MultiPack` each time.
        * ...
    - :class:`BaseProcessor`
    - :class:`BaseBatchProcessor`
    - ...



Users can refer to the full processor below.

.. code-block:: python

    class MachineTranslationProcessor(PackProcessor):
        """
        Translate the input text and output to a file.
        """
        def initialize(self, resources: Resources, configs: Config):
            super().initialize(resources, configs)

            # Initialize the tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token


            # Initialize the model
            model_name: str = self.configs.pretrained_model
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.task_prefix = "translate English to German: "


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




We also have plenty of written reader available to use. If you don't find one suitable in your case, you can refer to this documentation, API or tutorials to customize a new processor.
