import json
from typing import List
from forte.utils import create_import_error_msg
from forte.models.srl_new.data import RawExample, Example, SRLSpan

try:
    import torch
except ImportError as e:
    raise ImportError(
        create_import_error_msg("torch", "models", "models")
    ) from e


try:
    import texar.torch as tx
except ImportError as e:
    raise ImportError(
        create_import_error_msg(
            "texar-pytorch", "models", "Texar model support"
        )
    ) from e


class SRLSpanData(tx.data.DatasetBase[str, Example]):
    def __init__(self, path: str, vocab: tx.data.Vocab, hparams):
        source = tx.data.TextLineDataSource(path)
        self._vocab = vocab
        super().__init__(source, hparams)

    def process(self, raw_example: str) -> Example:
        raw: RawExample = json.loads(raw_example)
        assert len(raw["sentences"]) == 1
        sentence = raw["sentences"][0]
        example: Example = {
            "text": sentence,
            "text_ids": self._vocab.map_tokens_to_ids_py(sentence),
            "srl": [SRLSpan(*items) for items in raw["srl"][0]],
        }
        return example

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        sentences = [ex["text"] for ex in examples]
        tokens, length = tx.data.padded_batch(
            [ex["text_ids"] for ex in examples]
        )
        srl = [ex["srl"] for ex in examples]
        return tx.data.Batch(
            len(examples),
            srl=srl,
            text=sentences,
            text_ids=torch.from_numpy(tokens).to(self.device),
            length=torch.tensor(length).to(self.device),
        )
