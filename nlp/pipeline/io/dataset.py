import logging
import numpy as np
from typing import Iterator, Dict, List, Union, Set, Tuple, Optional
from nlp.pipeline.io.data_pack import DataPack
from nlp.pipeline.io.base_ontology import BaseOntology
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Dataset:
    """
    Example:
        .. code-block:: python
            reader = OntonotesReader()
            data = reader.dataset_iterator("conll-formatted-ontonotes-5.0")
            dataset = DatasetBase(data)

            annotype = {
                        "Token": ["pos_tag", "sense"],
                        "EntityMention": []
            }
            linktype = {
                "PredicateLink": ["parent", "parent.pred_lemma",
                                  "child", "arg_type"]
            }

            dataset.config_data_iterator(context_type="sentence",
                                         annotation_types = annotype,
                                         link_types = linktype)

            ## Three ways to get data:
            ## 1. get a piece of data
            sentence = dataset.get_data()
            ## 2. get a batch of data
            batch = dataset.get_data_batch(50)
            ## 3. iterate through the dataset
            for sentence in dataset.iterator:
                process(sentence)

    """

    def __init__(self, dataset):
        self.dataset: Iterator[DataPack] = dataset
        self.iterator: Optional[Iterator[Dict]] = None

    def config_data_iterator(
            self,
            context_type: str,
            annotation_types: Dict[str, Union[Dict, List, Set, Tuple]],
            link_types: Dict[str, Union[Dict, List, Set, Tuple]] = None,
            group_types: Dict[str, Union[Dict, List, Set, Tuple]] = None,
    ) -> None:

        self.iterator = self.data_iterator(context_type, annotation_types,
                                           link_types, group_types)

    def data_iterator(
            self,
            context_type: str,
            annotation_types: Dict[str, Union[Dict, List, Set, Tuple]],
            link_types: Dict[str, Union[Dict, List, Set, Tuple]] = None,
            group_types: Dict[str, Union[Dict, List, Set, Tuple]] = None,
    ) -> Iterator[Dict]:
        """
        Request one piece of data from the data set.

        Args:
            context_type (str): The granularity of the data context, which
                could be either `"sentence"` or `"document"`
            annotation_types (dict): The annotation types and fields required.
                The keys of the dict are the required annotation types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.
            link_types (dict): The link types and fields required.
                The keys of the dict are the required link types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.
            group_types (dict): The group types and fields required.
                The keys of the dict are the required group types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.

        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required annotations and context).
        """

        if context_type == "document":
            for doc in self.dataset:
                data = dict()
                data["context"] = doc.text
                for a_type, a_args in annotation_types.items():
                    data[a_type] = self._generate_span_annotation_data(
                        a_type, a_args, doc, None
                    )

                for a_type, a_args in link_types.items():
                    data[a_type] = self._generate_link_annotation_data(
                        a_type, a_args, doc, None
                    )
                yield data

        elif context_type == "sentence":

            sent_component = None
            sent_fields = {}
            sent_args = annotation_types.get("Sentence")
            if isinstance(sent_args, dict):
                sent_component = sent_args.get("component")
                sent_args = sent_args.get("fields", {})

            if isinstance(sent_args, (list, set, tuple)):
                sent_fields = set(sent_args)
            elif sent_args is not None:
                raise TypeError(
                    f"Invalid request for 'Sentence'. "
                    f"The request should be a list or a dict"
                )

            for doc in self.dataset:  # handle stop

                sent_meta = doc.internal_metas.get("Sentence")
                if sent_meta is None:
                    raise AttributeError(
                        f"Document '{doc.meta.docid}' has no sentence "
                        f"annotations'"
                    )

                if sent_component is None:
                    sent_component = sent_meta.default_component

                if sent_component not in sent_meta.fields_created.keys():
                    raise AttributeError(
                        f"Document '{doc.meta.docid}' has no sentence "
                        f"annotations generated by '{sent_component}'"
                    )

                valid_sent_ids = (doc.index.type_index["Sentence"]
                                  & doc.index.component_index[sent_component])

                for sent in doc.annotations:
                    if sent.tid not in valid_sent_ids:
                        continue

                    data = dict()
                    data["context"] = doc.text[sent.span.begin: sent.span.end]

                    for f in sent_fields:
                        if f not in sent_meta.fields_created[sent_component]:
                            raise AttributeError(
                                f"Sentence annotation generated by"
                                f" '{sent_component}' has no field named '{f}'"
                            )

                        data[f] = getattr(sent, f)

                    for a_type, a_args in annotation_types.items():
                        if a_type == "Sentence":
                            continue

                        data[a_type] = self._generate_span_annotation_data(
                            a_type, a_args, doc, sent
                        )

                    for a_type, a_args in link_types.items():
                        data[a_type] = self._generate_link_annotation_data(
                            a_type, a_args, doc, sent
                        )

                    yield data

    @staticmethod
    def _process_request_args(a_type, a_args, doc):

        # check the existence of ``a_type`` annotation in ``doc``
        a_meta = doc.internal_metas.get(a_type)
        if a_meta is None:
            raise AttributeError(
                f"Document '{doc.meta.docid}' has no '{a_type}' "
                f"annotations'"
            )

        # request which fields from which component
        component = None
        if isinstance(a_args, dict):
            component = a_args.get("component")
            a_args = a_args.get("fields", {})

        if isinstance(a_args, (list, set, tuple)):
            fields = set(a_args)
        else:
            raise TypeError(
                f"Invalid request for '{a_type}'. The request"
                f" should be a list or a dict"
            )

        if component is None:
            component = a_meta.default_component

        if component not in a_meta.fields_created.keys():
            raise AttributeError(
                f"DataPack has no {a_type} annotations generated"
                f" by {component}"
            )

        return component, fields

    def _generate_span_annotation_data(
            self,
            a_type: str,
            a_args: Union[Dict, List],
            doc: DataPack,
            sent: Optional[BaseOntology.Sentence]) -> Dict:

        component, fields = self._process_request_args(a_type, a_args, doc)

        a_dict = defaultdict(list)
        sent_begin = sent.span.begin if sent else 0

        # ``a_type`` annotations generated by ``component`` in this ``sent``

        valid_annotation = (doc.index.type_index[a_type]
                            & doc.index.component_index[component])
        if sent:
            valid_annotation &= doc.index.sentence_index[sent.tid]

        for annotation in doc.annotations:
            if annotation.tid not in valid_annotation:
                continue
            a_dict["span"].append((annotation.span.begin - sent_begin,
                                   annotation.span.end - sent_begin))
            a_dict["text"].append(doc.text[annotation.span.begin:
                                           annotation.span.end])
            for f in fields:
                if f not in doc.internal_metas[a_type].fields_created[component]:
                    raise AttributeError(
                        f"'{a_type}' annotation generated by "
                        f"'{component}' has no field named '{f}'"
                    )
                a_dict[f].append(getattr(annotation, f))

        for k, v in a_dict.items():
            a_dict[k] = np.array(v)
        return a_dict

    def _generate_link_annotation_data(
            self,
            a_type: str,
            a_args: Union[Dict, List],
            doc: DataPack,
            sent: Optional[BaseOntology.Sentence],
    ) -> Dict:

        component, fields = self._process_request_args(a_type, a_args, doc)

        a_dict = defaultdict(list)
        sent_begin = sent.span.begin if sent else 0

        # ``a_type`` annotations generated by ``component`` in this ``sent``
        valid_annotation = (doc.index.type_index[a_type]
                            & doc.index.component_index[component])

        if sent:
            valid_annotation &= doc.index.sentence_index[sent.tid]

        for a_id in sorted(valid_annotation):  # in tid string order
            annotation = doc.index.entry_index[a_id]

            parent_fields = {f for f in fields if f.split('.')[0] == "parent"}
            if parent_fields:
                p_id = annotation.parent
                parent = doc.index.entry_index[p_id]
                p_type = parent.__class__.__name__
                a_dict["parent.span"].append((parent.span.begin - sent_begin,
                                              parent.span.end - sent_begin,))
                a_dict["parent.text"].append(doc.text[parent.span.begin:
                                                      parent.span.end])
                for f in parent_fields:
                    pf = f.split(".")
                    if len(pf) == 1:
                        continue
                    if len(pf) > 2:
                        raise AttributeError(
                            f"Too many delimiters in field name {f}."
                        )
                    pf = pf[1]

                    if pf not in doc.internal_metas[p_type].fields_created[parent.component]:
                        raise AttributeError(
                            f"'{p_type}' annotation generated by "
                            f"'{parent.component}' has no field named '{pf}'"
                        )
                    a_dict[f].append(getattr(parent, pf))

            child_fields = {f for f in fields if f.split('.')[0] == "child"}
            if child_fields:
                c_id = annotation.child
                child = doc.index.entry_index[c_id]
                c_type = child.__class__.__name__

                a_dict["child.span"].append((child.span.begin - sent_begin,
                                             child.span.end - sent_begin))
                a_dict["child.text"].append(doc.text[child.span.begin:
                                                     child.span.end])
                for f in child_fields:
                    cf = f.split(".")
                    if len(cf) == 1:
                        continue
                    if len(cf) > 2:
                        raise AttributeError(
                            f"Too many delimiters in field name {f}."
                        )
                    cf = cf[1]

                    if cf not in doc.internal_metas[c_type].fields_created[child.component]:
                        raise AttributeError(
                            f"'{c_type}' annotation generated by "
                            f"'{child.component}' has no field named '{cf}'"
                        )
                    a_dict[f].append(getattr(child, cf))

            for f in (fields - parent_fields - child_fields):
                if f not in doc.internal_metas[a_type].fields_created[component]:
                    raise AttributeError(
                        f"'{a_type}' annotation generated by "
                        f"'{component}' has no field named '{f}'"
                    )
                a_dict[f].append(getattr(annotation, f))

        for k, v in a_dict.items():
            a_dict[k] = np.array(v)
        return a_dict

    def get_data(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return None

    def get_data_batch(self, batch_size: int):

        batch = {}
        for i in range(batch_size):
            data = self.get_data()
            if data is None: break
            for entry, fields in data.items():
                if isinstance(fields, dict):
                    if entry not in batch.keys():
                        batch[entry] = {}
                    for k, v in fields.items():
                        if k not in batch[entry].keys():
                            batch[entry][k] = []
                        batch[entry][k].append(v)
                else:  # context level feature
                    if entry not in batch.keys():
                        batch[entry] = []
                    batch[entry].append(fields)

        if batch:
            return batch
        else:
            return None
