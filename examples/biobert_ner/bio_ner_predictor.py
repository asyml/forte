# pylint: disable=logging-fstring-interpolation
from typing import Dict, List, Optional, Type

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.types import DataRequest
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from ft.onto.base_ontology import Sentence, EntityMention, Subword


class BioBERTNERPredictor(FixedSizeBatchProcessor):
    """
       An Named Entity Recognizer fine-tuned on BioBERT

       Note that to use :class:`BioBERTNERPredictor`, the :attr:`ontology` of
       :class:`Pipeline` must be an ontology that include
       ``ft.onto.base_ontology.Subword`` and ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()
        self.resources = None
        self.device = None

        self.ft_configs = None
        self.model_config = None
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _define_context() -> Type[Annotation]:
        return Sentence

    @staticmethod
    def _define_input_info() -> DataRequest:
        input_info: DataRequest = {
            Subword: [],
            Sentence: [],
        }
        return input_info

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if resources.get("device"):
            self.device = resources.get("device")
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')

        self.resources = resources
        self.ft_configs = configs

        model_path = self.ft_configs.model_path
        self.model_config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=self.model_config
        )
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, data_batch: Dict[str, Dict[str, List[str]]]) \
            -> Dict[str, Dict[str, List[np.array]]]:
        sentences = data_batch['context']
        subwords = data_batch['Subword']

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        input_shape = inputs['input_ids'].shape
        if input_shape[1] > 512:
            # TODO: Temporarily work around the length problem.
            #   The real solution should further split up the sentences to make
            #   the sentences shorter.
            labels_idx = inputs['input_ids'].new_full(
                input_shape, 2, device='cpu')[:, 1:-1].numpy()
        else:
            outputs = self.model(**inputs)[0].cpu().numpy()
            score = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)[:, 1:-1]  # Remove placeholders.

        pred: Dict = {"Subword": {"ner": [], "tid": []}}

        for i in range(len(subwords["tid"])):
            tids = subwords["tid"][i]
            ner_tags = []
            for j in range(len(tids)):
                ner_tags.append(self.model.config.id2label[labels_idx[i, j]])

            pred["Subword"]["ner"].append(np.array(ner_tags))
            pred["Subword"]["tid"].append(np.array(tids))

        return pred

    def _fill_sub_entities(self, entity_groups_disagg, data_pack, tids):
        first_idx = entity_groups_disagg[0]['idx']
        first_tid = entity_groups_disagg[0]['tid']
        while first_idx > 0 and data_pack.get_entry(first_tid).is_subword:
            first_idx -= 1
            first_tid -= 1

        last_idx = entity_groups_disagg[-1]['idx']
        last_tid = entity_groups_disagg[-1]['tid']

        while last_idx < len(tids) and data_pack.get_entry(last_tid + 1) \
                .is_subword:
            last_idx += 1
            last_tid += 1

        return first_tid, last_tid

    def _group_entities(self, entities, data_pack, tids):
        """Find and group adjacent tokens considered to have the same entity

        Logic: An entity starts with a "B" and extends to the end of the
        word that contains last "I"

        """
        entity_groups = []
        entity_groups_disagg = []

        for entity in entities:
            subword = data_pack.get_entry(entity['tid'])
            if entity['label'] == 'B' and not subword.is_subword:
                if entity_groups_disagg:
                    entity_groups_disagg = \
                        self._fill_sub_entities(entity_groups_disagg,
                                                data_pack,
                                                tids)
                    entity_groups.append(entity_groups_disagg)
                entity_groups_disagg = [entity]
            else:
                entity_groups_disagg.append(entity)

        if entity_groups_disagg:
            entity_groups_disagg = self._fill_sub_entities(entity_groups_disagg,
                                                           data_pack,
                                                           tids)
            entity_groups.append(entity_groups_disagg)

        return entity_groups

    def pack(self, data_pack: DataPack,
             output_dict: Optional[Dict[str, Dict[str, List[str]]]] = None):
        """
        Write the prediction results back to datapack. by writing the predicted
        ner to the original subwords and convert predictions to something that
        makes sense in a word-by-word segmentation
        """

        if output_dict is None:
            return

        for i in range(len(output_dict["Subword"]["tid"])):
            tids = output_dict["Subword"]["tid"][i]
            labels = output_dict["Subword"]["ner"][i]

            entities = []
            # Filter to labels not in `self.ignore_labels`
            entities = [dict(idx=idx, label=label, tid=tid)
                        for idx, (label, tid) in enumerate(zip(labels, tids))
                        if label not in self.ft_configs.ignore_labels]

            entity_groups = self._group_entities(entities, data_pack, tids)
            # Add NER tags and create EntityMention ontologies.
            for first_tid, last_tid in entity_groups:
                first_token: Subword = data_pack.get_entry(  # type: ignore
                    first_tid)
                first_token.ner = 'B-' + self.ft_configs.ner_type

                for tid in range(first_tid + 1, last_tid + 1):
                    token: Subword = data_pack.get_entry(tid)  # type: ignore
                    token.ner = 'I-' + self.ft_configs.ner_type

                begin = first_token.span.begin
                end = data_pack.get_entry(last_tid).span.end
                entity = EntityMention(data_pack, begin, end)
                entity.ner_type = self.ft_configs.ner_type

    @classmethod
    def default_configs(cls):
        r"""Default config for NER Predictor"""

        configs = super().default_configs()
        # TODO: Batcher in NER need to be update to use the sytem one.
        configs["batcher"] = {"batch_size": 10}

        more_configs = {'model_path': None,
                        'ner_type': 'BioEntity',
                        'ignore_labels': ['O']}

        configs.update(more_configs)
        return configs
