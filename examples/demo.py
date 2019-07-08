from nlp.pipeline import get_full_component_name
from nlp.pipeline.processors.dummy_processor import RelationOntology as Ont
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors import (
    NLTKSentenceSegmenter, NLTKWordTokenizer,
    NLTKPOSTagger, CoNLLNERPredictor, SRLPredictor,
)


def main():
    # kwargs = {
    #     "dataset": {
    #         "dataset_dir": "./bbc",
    #         "dataset_format": "plain"
    #     }
    # }
    kwargs = {
        "dataset": {
            "dataset_dir": "./ontonotes_sample_dataset",
            "dataset_format": "ontonotes",
        }
    }

    pl = Pipeline(**kwargs)
    # pl.processors.append(NLTKSentenceSegmenter())
    # pl.processors.append(NLTKWordTokenizer())
    # pl.processors.append(NLTKPOSTagger())
    # pl.processors.append(CoNLLNERPredictor())
    pl.processors.append(SRLPredictor(model_dir="../../srl/texar-srl"))

    for pack in pl.run():
        print(pack.meta.doc_id)
        print(pack.text)
        for sentence in pack.get(Ont.Sentence):
            sent_text = sentence.text
            print(sent_text)
            # first method to get entry in a sentence
            for link in pack.get(
                    Ont.PredicateLink, sentence,
                    component=pl.processors[0].component_name):
                parent = link.get_parent()
                child = link.get_child()
                print(f"SRL: \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")

            # second method to get entry in a sentence
            tokens = [(token.text, token.pos_tag) for token in
                      pack.get(Ont.Token, sentence)]
            print("Tokens:", tokens)
        input("Press ENTER to continue...")


if __name__ == '__main__':
    main()
