import json
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
from mypy_extensions import TypedDict
import texar.torch as tx


class SRLSpan(NamedTuple):
    predicate: int
    start: int
    end: int
    label: str


class Span(NamedTuple):
    start: int
    end: int
    label: str


class RawExample(TypedDict):
    speakers: List[List[str]]
    doc_key: str
    sentences: List[List[str]]
    srl: List[List[Tuple[int, int, int, str]]]
    constituents: List[List[Span]]
    cluster: List
    ner: List[List[Span]]


class Example(TypedDict):
    text: List[str]
    text_ids: np.ndarray
    srl: List[SRLSpan]


class SRLSpanData(tx.data.DataBase[str, Example]):
    def __init__(self, path: str, vocab: tx.data.Vocab, hparams):
        source = tx.data.TextLineDataSource(path)
        self._vocab = vocab
        super().__init__(source, hparams)

    def process(self, raw_example: str) -> Example:
        raw: RawExample = json.loads(raw_example)
        assert len(raw['sentences']) == 1
        sentence = raw['sentences'][0]
        example: Example = {
            'text': sentence,
            'text_ids': self._vocab.map_tokens_to_ids_py(sentence),
            'srl': [SRLSpan(*items) for items in raw['srl'][0]]
        }
        return example

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        sentences = [ex['text'] for ex in examples]
        tokens, length = tx.data.padded_batch(
            [ex['text_ids'] for ex in examples])
        srl = [ex['srl'] for ex in examples]
        return tx.data.Batch(
            len(examples), srl=srl, text=sentences,
            text_ids=torch.from_numpy(tokens).to(self.device),
            length=torch.tensor(length).to(self.device))
