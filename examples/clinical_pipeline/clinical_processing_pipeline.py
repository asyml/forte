from mimic3_note_reader import Mimic3DischargeNoteReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKNER

# Prepare the data.

pl = Pipeline[DataPack]()
pl.set_reader(Mimic3DischargeNoteReader())
pl.add(NLTKSentenceSegmenter())
pl.add(NLTKWordTokenizer())
pl.add(NLTKPOSTagger())
pl.add(NLTKNER())

# Index the dataset

