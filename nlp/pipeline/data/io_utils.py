from typing import Dict, List

__all__ = [
    "batch_instances",
    "merge_batches",
    "slice_batch",
]


def batch_instances(instances: List[Dict]):
    """
    Merge a list of ``instances``.
    """
    batch: Dict = {}
    for instance in instances:
        for entry, fields in instance.items():
            if isinstance(fields, dict):
                if entry not in batch.keys():
                    batch[entry] = {}
                for k, value in fields.items():
                    if k not in batch[entry].keys():
                        batch[entry][k] = []
                    batch[entry][k].append(value)
            else:  # context level feature
                if entry not in batch.keys():
                    batch[entry] = []
                batch[entry].append(fields)
    return batch


def merge_batches(batches: List[Dict]):
    """
    Merge a list of or ``batches``.
    """
    merged_batch: Dict = {}
    for batch in batches:
        for entry, fields in batch.items():
            if isinstance(fields, dict):
                if entry not in merged_batch.keys():
                    merged_batch[entry] = {}
                for k, value in fields.items():
                    if k not in merged_batch[entry].keys():
                        merged_batch[entry][k] = []
                    merged_batch[entry][k].extend(value)
            else:  # context level feature
                if entry not in merged_batch.keys():
                    merged_batch[entry] = []
                merged_batch[entry].extend(fields)
    return merged_batch


def slice_batch(batch, start, length):
    """
    Return a sliced batch of size ``length`` from ``start`` in ``batch``.
    """
    sliced_batch: Dict = {}

    for entry, fields in batch.items():
        if isinstance(fields, dict):
            if entry not in sliced_batch.keys():
                sliced_batch[entry] = {}
            for k, value in fields.items():
                sliced_batch[entry][k] = value[start: start + length]
        else:  # context level feature
            sliced_batch[entry] = fields[start: start + length]

    return sliced_batch
