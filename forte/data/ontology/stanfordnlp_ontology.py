import forte.data.ontology.base_ontology as ontology


class Word(ontology.Token):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.dependency_relation = None
        self.feats = None
        self.governor = None
        self.index = None
        self.lemma = None
        self.pos = None
        self.text_ = None
        self.upos = None
        self.xpos = None
        self.pos_tag = None


class Token(ontology.Token):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.index = None
        self.text_ = None
        self.words = None


class Sentence(ontology.Sentence):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.dependencies = None
        self.tokens = None
        self.words = None

class Document(ontology.Document):
    def __init__(self):
        super().__init__()
        self.text_ = None