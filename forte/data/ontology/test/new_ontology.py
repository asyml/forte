import forte.data.ontology
import forte.data.ontology.top


class Token(forte.data.ontology.top.Annotation):
	def __init__(self, *args):
		super().__init__(args)
		self.pos_tag: Optional[str] = None
		self.lemma: Optional[str] = None


class EntityMention(forte.data.ontology.top.Annotation):
	def __init__(self, *args):
		super().__init__(args)
		self.entity_type: Optional[str] = None
