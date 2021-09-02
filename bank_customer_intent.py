from forte import Pipeline
from forte.data.readers import Banking77
from forte.spacy import SpacyProcessor
from forte.huggingface import ZeroShotClassifier
from termcolor import colored
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.common.configuration import Config
from forte.nltk import NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence, Token, Document


csv_path = "/Users/pengfei.he/Downloads/banking77/test.csv"
def get_model_config(reader, model):
    """
    Get model configuration with updating from reader configuration.
    :return:
    """
    def update_config_labels():
        """
        Update model config label names from reader config's label names
        :return:
        """
        model_config['candidate_labels'] = list(reader_config['class_names'].values())
    reader_config = reader.default_configs()  # including classification info
    model_config = model.default_configs()
    update_config_labels()
    return model_config

pl = Pipeline()
pl.set_reader(Banking77())
pl.add(NLTKSentenceSegmenter())
pl.add(NLTKWordTokenizer())
model_config = get_model_config(Banking77(), ZeroShotClassifier())
pl.add(ZeroShotClassifier(), config=model_config)
pl.initialize()



for pack in pl.process_dataset(csv_path):
    for sentence in pack.get(Sentence):
        import pdb; pdb.set_trace()
        sent_text = sentence.text
        print(colored("Sentence:", "red"), sent_text, "\n")
        print(colored("Prediction:", "blue"), sentence.classification)
        # first method to get entry in a sentence
        tokens = [
            (token.text, token.pos) for token in pack.get(Token, sentence)
        ]





