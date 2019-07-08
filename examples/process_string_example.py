from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors import CoNLL03Ontology as Ont
from nlp.pipeline.processors import (
    NLTKSentenceSegmenter, NLTKWordTokenizer,
    NLTKPOSTagger, CoNLLNERPredictor
)

text = "The plain green Norway spruce is displayed in the gallery's foyer. " \
       "Wentworth worked as an assistant to sculptor Henry Moore in the " \
       "late 1960s. His reputation as a sculptor grew in the 1980s." \

pl = Pipeline()
pl.processors.append(NLTKSentenceSegmenter())
pl.processors.append(NLTKWordTokenizer())
pl.processors.append(NLTKPOSTagger())
# pl.processors.append(CoNLLNERPredictor())

pack = pl.process(text)

for sentence in pack.get(Ont.Sentence):
    sent_text = sentence.text
    print("Sentence:", sent_text)
    # first method to get entry in a sentence
    for link in pack.get(Ont.PredicateLink, sentence):
        parent = link.get_parent()
        child = link.get_child()
        print(f"Links: {parent.text} is {link.rel_type} {child.text}")

    # second method to get entry in a sentence
    tokens = [(token.text, token.pos_tag) for token in
              pack.get(Ont.Token, sentence)]
    print("Tokens:", tokens)

    input()

