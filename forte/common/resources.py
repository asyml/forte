import os
import pickle

__all__ = [
    "Resources",
]


class Resources:
    """
    The Resources object is a global registry used in the pipeline
    The objects defined as the ``Resources`` will be passed on to the
        processors in the pipeline for the initialization
    """
    def __init__(self, **kwargs):
        self.resources = {}
        self.update(**kwargs)

    def save(self, output_dir="./"):
        with open(os.path.join(output_dir, "resources.pkl"), "wb") as output:
            pickle.dump(self.resources, output, pickle.HIGHEST_PROTOCOL)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            try:
                pickle.dumps(value)
            # pylint: disable=broad-except
            except Exception as e:
                print(f'Value:{value} cannot be pickled. {e}')
            self.resources[key] = value

    def load(self, path):
        resources = pickle.load(open(path, 'rb'))
        self.update(**resources)
