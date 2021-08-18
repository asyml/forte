from forte.huggingface import ZeroShotClassifier
from forte.stanza import StandfordNLPProcessor

from forte import Pipeline
from forte.data.readers import TerminalReader
from forte.processors.stave import StaveProcessor

nlp = Pipeline()
nlp.set_reader(TerminalReader())
nlp.add(StandfordNLPProcessor())
nlp.add(
    ZeroShotClassifier(),
    config={
        "candidate_labels": [
            "travel",
            "cooking",
            "dancing",
            "exploration",
        ],
    },
)
nlp.add(StaveProcessor())
nlp.initialize()
nlp.run()
