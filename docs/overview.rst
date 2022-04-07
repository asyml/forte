
Introduction
----------------
**Forte** is a toolkit for building Natural Language Processing pipelines, featuring cross-task
interaction, adaptable data-model interfaces and many more. It provides a platform to assemble
state-of-the-art NLP and ML technologies in a highly-composable fashion, including a wide
spectrum of tasks ranging from Information Retrieval, Natural Language Understanding to Natural
Language Generation.

With Forte, it is extremely simple to build an integrated system that can search documents,
analyze and extract information and generate language all in one place. This allows the developer
to fully utilize and combine the strength and results from each step, and allow the system to
make fully informed decision at the end of the pipeline.

While it is quite easy to combine arbitrary 3rd party tools (Check out these `examples <index_appendices.html>`_ !),
Forte also brings technology to you by supporting deep learning via Texar, and by providing a convenient
model data interface that allows user to cast tasks to models.


Download and Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forte is available through PIP by 
::

  pip install forte


You can get our latest code through source.
::

  git clone https://github.com/asyml/forte.git
  cd forte
  pip install .




Library API example
--------------------
A simple code example that runs Named Entity Recognizer

.. code-block:: python

   import yaml

   from forte.pipeline import Pipeline
   from forte.data.readers import CoNLL03Reader
   from forte.processors.nlp import CoNLLNERPredictor
   from ft.onto.base_ontology import Token, Sentence
   from forte.common.configuration import Config


   config_data = yaml.safe_load(open("config_data.yml", "r"))
   config_model = yaml.safe_load(open("config_model.yml", "r"))

   config = Config({}, default_hparams=None)
   config.add_hparam('config_data', config_data)
   config.add_hparam('config_model', config_model)


   pl = Pipeline()
   pl.set_reader(CoNLL03Reader())
   pl.add(CoNLLNERPredictor(), config=config)

   pl.initialize()

   for pack in pl.process_dataset(config.config_data.test_path):
      for pred_sentence in pack.get_data(context_type=Sentence, request={Token: {"fields": ["ner"]}}):
         print("============================")
         print(pred_sentence["context"])
         print("The entities are...")
         print(pred_sentence["Token"]["ner"])
         print("============================")



Many more examples are available `here  <index_appendices.html>`_. We are also working assembling some
interesting `tutorials <https://github.com/asyml/forte/wiki>`_


License
~~~~~~~~~

`Apache License 2.0 <https://github.com/asyml/forte/blob/master/LICENSE>`_
