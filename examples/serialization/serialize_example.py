from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import OntonotesReader
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter
from forte.processors.writers import DocIdJsonPackWriter

nlp = Pipeline()
reader = OntonotesReader()

data_path = "../data_samples/ontonotes/00/"

nlp.set_reader(OntonotesReader())
nlp.add_processor(NLTKSentenceSegmenter())
nlp.add_processor(NLTKWordTokenizer())
nlp.add_processor(NLTKPOSTagger())

# This is a simple writer that serialize the result to the current directory and
# will use the DocID field in the data pack as the file name.
nlp.add_processor(DocIdJsonPackWriter(), HParams(
    {
        'output_dir': '.'
    },
    DocIdJsonPackWriter.default_hparams(),
))

nlp.initialize()

nlp.run(data_path)
