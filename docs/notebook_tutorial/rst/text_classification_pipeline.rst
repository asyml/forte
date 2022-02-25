Text Classification Pipeline
============================

Packages
--------

.. code:: python

    import sys
    from termcolor import colored
    from forte.data.readers import ClassificationDatasetReader
    from fortex.huggingface import ZeroShotClassifier
    from forte.pipeline import Pipeline
    from fortex.nltk import NLTKSentenceSegmenter
    from ft.onto.base_ontology import Sentence

Background
----------

This notebook tutorial is derived from `a classification
example <https://github.com/asyml/forte/tree/master/examples/classification>`__.
Given a table-like csv file with data at some columns are input text and
data at one column is label, we set up a text classification pipeline
below. This example is also a good example of wrapping external library
classes/methods into ``PipelineComponent``.

Inference Workflow
------------------

Pipeline
~~~~~~~~

-  `Pipeline
   setup <https://github.com/asyml/forte/blob/master/examples/classification/bank_customer_intent.py#L123>`__

-  The pipeline has one reader ``ClassificationDatasetReader`` and two
   processor ``NLTKSentenceSegmenter`` and ``ZeroShotClassifier``.

Reader
~~~~~~

-  `ClassificationDatasetReader <https://github.com/asyml/forte/blob/7dc6e6c7d62d9a4126bdfc5ca02d15be3ffd61ca/forte/data/readers/classification_reader.py#L26>`__

   -  ``set_up()``: It checks whether the configuration is correct. For
      example, ``skip_k_starting_lines`` should be larger than 0
      otherwise it doesn’t make sense. It also converts different table
      data at the label column to a digit.
   -  ``_collect()``: read rows from csv file and returns iterator that
      yields line id and line data.
   -  ``_cache_key_function()``: use the line id as the cache key.
   -  ``_parse_pack()``: parse data from iterator returned by
      ``_collect`` and load it in the datapack

Processor
~~~~~~~~~

In this example, we want to classify data sentence by sentence so we
wrapped ``nltk.PunktSentenceTokenizer`` in
`NLTKSentenceSegmenter <https://github.com/asyml/forte-wrappers/blob/80cfe19926c0596edd13985581e8ca01a7be86ad/src/nltk/fortex/nltk/nltk_processors.py#L247>`__
to segment sentences.

-  ``_process()``: split data pack text into sentence spans.

Then need a model to do classification. We wrap
``transformers.pipeline`` in `Huggingface
ZeroShotClassifier <https://github.com/asyml/forte-wrappers/blob/main/src/huggingface/fortex/huggingface/zero_shot_classifier.py>`__.

-  ``_process()``: running classifier over data pack data and write the
   prediction results back to data pack.

``ZeroShotClassifier`` and ``NLTKSentenceSegmenter`` both inherit from
``PackProcessor`` as it processes one ``DataPack`` at a time. Suppose if
we processes one ``MultiPack`` at a time, we need to inherit
``MultiPackProcessor`` instead.

.. code:: python


    csv_path = "data_samples/amazon_review_polarity_csv/sample.csv"
    pl = Pipeline()

    # initialize labels
    class_names = ["negative", "positive"]
    index2class = dict(enumerate(class_names))
    pl.set_reader(
        ClassificationDatasetReader(), config={"index2class": index2class}
    )
    pl.add(NLTKSentenceSegmenter())
    pl.add(ZeroShotClassifier(), config={"candidate_labels": class_names})
    pl.initialize()


    for pack in pl.process_dataset(csv_path):
        for sent in pack.get(Sentence):
            if (
                input("Type n for the next documentation and its prediction: ").lower()
                == "n"
            ):
                sent_text = sent.text
                print(colored("Sentence:", "red"), sent_text, "\n")
                print(colored("Prediction:", "blue"), sent.classification)
            else:
                print("Exit the program due to unrecognized input")
                sys.exit()
