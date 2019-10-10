from pathlib import Path
import pickle
from typing import Union, Dict, List, Callable, Optional, Any
from collections.abc import KeysView

__all__ = [
    "Resources"
]

SerializeDict = Dict[str, Callable[[Any, Union[str, Path]], None]]
DeserializeDict = Dict[str, Callable[[Union[str, Path]], None]]


class Resources:
    r"""The Resources object is a global registry used in the pipeline. Objects
    defined as the ``Resources`` will be passed on to the processors in the
    pipeline for the initialization.
    """

    def __init__(self, **kwargs):
        self.resources = {}
        self.update(**kwargs)

    def save(self, keys: Optional[Union[List[str], SerializeDict]] = None,
             output_dir: Optional[str] = None):
        r"""Save the resources specified by `keys` in binary format.

        Args:
            keys (optional): list or dict
                - If `keys` is a list, the objects corresponding to those keys
                are saved
                - If `keys` is a dict mapping from a key to a serialize
                function, then the serialize function will be used to save the
                object corresponding to that key
                - If `keys` is None, all objects in this resource will be saved.
            output_dir (optional): str
                A directory specifying the location to save the resources.
        """

        # TODO: use a default save directory like default_save_dir() if None
        if output_dir is None:
            output_dir = "./"

        if keys is None:
            keys = self.resources.keys()

        if isinstance(keys, List):
            for key in keys:
                with open(Path(output_dir, f"{key}.pkl"), "wb") as f:
                    pickle.dump(self.resources.get(key), f,
                                pickle.HIGHEST_PROTOCOL)
        else:
            for key, serializer in keys.items():
                serializer(self.resources[key], Path(output_dir, f"{key}.pkl"))

    def keys(self) -> KeysView:
        return self.resources.keys()

    def get(self, key: str):
        return self.resources.get(key)

    def update(self, **kwargs):
        self.resources.update(**kwargs)

    def remove(self, key: str):
        del self.resources[key]

    def load(self, keys: Union[List[str], DeserializeDict],
             path: Optional[str] = None):
        r"""Load the resources specified by `keys`.

        Args:
            keys: list or dict
                - If `keys` is a list, the objects corresponding to those keys
                are loaded
                - If `keys` is a dict mapping from a key to a deserialize
                function, then the deserialize function will be used to load the
                object corresponding to that key
            path (optional): str
                A directory specifying the location to load the resources from.
        """

        # TODO: use a default save directory like default_save_dir() if None
        if path is None:
            path = "./"

        if isinstance(keys, List):
            for key in keys:
                with open(Path(path, f"{key}.pkl"), "rb") as f:
                    self.resources[key] = pickle.load(f)

        else:
            for key, deserializer in keys.items():
                self.resources[key] = deserializer(Path(path, f"{key}.pkl"))
