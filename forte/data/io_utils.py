import os
from typing import Dict, List, Iterator, Tuple

from forte.data.ontology import Span
from forte.data.data_pack import ReplaceOperationsType

__all__ = [
    "batch_instances",
    "merge_batches",
    "slice_batch",
    "dataset_path_iterator"
]


def batch_instances(instances: List[Dict]):
    """
    Merge a list of instances.
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
    Merge a list of instances or batches.
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


def dataset_path_iterator(dir_path: str, file_extension: str) -> Iterator[str]:
    """
    An iterator returning file_paths in a directory containing files
    of the given format
    """
    for root, _, files in os.walk(dir_path):
        for data_file in files:
            if len(file_extension) > 0:
                if data_file.endswith(file_extension):
                    yield os.path.join(root, data_file)
            else:
                yield os.path.join(root, data_file)


def modify_text_and_track_ops(text: str, span_ops: ReplaceOperationsType)\
        -> (str, ReplaceOperationsType, Dict[Span, Span], int):
    """
    Modifies the :param text using :param span_ops
    Assumes that span_ops are mutually exclusive
    :return mod_text: Modified text
            inverse_operations: ReplaceOperations to obtain original text back
            inverse_original_spans: List of replacement and original span
            orig_text_len: length of original text
    """
    orig_text_len: int = len(text)
    mod_text: str = text
    increment: int = 0
    prev_span_end: int = 0
    inverse_operations: List[Tuple[Span, str]] = []
    inverse_original_spans: List[Tuple[Span, Span]] = []

    # Sorting the spans such that the order of replacement strings
    # is maintained -> utilizing the stable sort property of python sort
    span_ops.sort(key=lambda item: item[0])

    for span, replacement in span_ops:
        if span.begin < 0 or span.end < 0:
            raise ValueError(
                "Negative indexing not supported")
        if span.begin > len(text) or span.end > len(text):
            raise ValueError(
                "One of the span indices are outside the string length")
        if span.end < span.begin:
            print(span.begin, span.end)
            raise ValueError(
                "One of the end indices is lesser than start index")
        if span.begin < prev_span_end:
            raise ValueError(
                "The replacement spans should be mutually exclusive")
        span_begin = span.begin + increment
        span_end = span.end + increment
        original_span_text = mod_text[span_begin: span_end]
        mod_text = mod_text[:span_begin] + replacement + mod_text[span_end:]
        increment += len(replacement) - (span.end - span.begin)
        replacement_span = Span(span_begin, span_begin + len(replacement))
        inverse_operations.append((replacement_span, original_span_text))
        inverse_original_spans.append((replacement_span, span))
        prev_span_end = span.end

        return mod_text, inverse_operations, sorted(inverse_original_spans), \
            orig_text_len
