"""
The Resources class wraps necessary resources to build a processor ( or a
trainer)
"""
import os
import pickle

__all__ = [
    "Resources",
]


class Resources:
    def __init__(self, **kwargs):
        self.resources = {}
        self.resources.update(kwargs)

    def save(self, output_dir="./"):
        with open(os.path.join(output_dir, "resources.pkl"), "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
