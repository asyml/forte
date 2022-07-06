from abc import abstractclassmethod
from typing import Callable
from forte.data.ontology.top import Meta
from functools import partial
import numpy as np


class PayloadFactory:
    """
    A class that handles the creation of payloads. It can register meta data and check its validity.
    """

    def __init__(self):
        self.valid_meta = (
            {}
        )  # map from meta data name to function that handles this meta data

    def register(self, meta: Meta):
        """
        A function that registers a meta data type into the factory.

        Args:
            meta_name: a Generic object that is used to register a Payload meta data type.
        """
        if meta.source_type not in ("web", "local"):
            raise ValueError("Meta data source must be either 'web' or 'local'")

        self.valid_meta[meta] = True

    def check_meta(self, meta):
        if meta in self.valid_meta:
            return True
        else:
            return False


class Payloading:
    """
    An class that help mapping meta data to loading function.
    """

    def __init__(self):
        self._factory = PayloadFactory()

    def load_factory(self, factory):
        self._factory = factory

    @abstractclassmethod
    def route(meta):
        """
        Convert the meta into a loading function that takes uri and read data
        from the uri.

        Args:
            meta: Meta data

        Returns:
            a function that takes uri and read data from the uri.
        """
        pass


class ImagePayloading(Payloading):
    """
    A class that helps mapping Image meta data to loading function.
    """

    def route(self, meta) -> Callable:
        """
        A function that takes and analyzes a meta data and returns a
        corresponding loading function.

        Args:
            meta: a Meta object that is used to determine the loading function.

        Raises:
            ValueError: if meta is not a valid meta data.

        Returns:
            _type_: _description_
        """
        if not self._factory.check_meta(meta):
            raise ValueError(f"Meta data{meta} not supported")
        if meta.source_type == "local":
            try:
                import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "ImagePayloading reading local file requires `matplotlib` package to be installed."
                ) from e
            return plt.imread
        elif meta.source_type == "web":
            try:
                from PIL import Image
                import requests
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "ImagePayloading reading web file requires `PIL` and `requests` packages to be installed."
                ) from e

            def read_uri(uri):
                # customize this function to read data from uri
                uri_obj = requests.get(uri, stream=True)
                pil_image = Image.open(uri_obj.raw)
                return self._pil_to_nparray(pil_image)

            return read_uri

    def _pil_to_nparray(self, pil_image):
        return np.asarray(pil_image)


class AudioPayloading(Payloading):
    def route(self, meta) -> Callable:
        """
        A function that

        Args:
            meta (_type_): _description_

        Returns:
            Callable: _description_
        """
        if meta.source_type == "local":
            try:
                import soundfile  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "AudioPayloading requires 'soundfile' package to be installed."
                    " You can refer to [extra modules to install]('pip install"
                    " forte['audio_ext']) or 'pip install forte"
                    ". Note that additional steps might apply to Linux"
                    " users (refer to "
                    "https://pysoundfile.readthedocs.io/en/latest/#installation)."
                ) from e

            def get_first(
                seq,
            ):  # takes the first item as soundfile returns a tuple of (data, samplerate)
                return seq[0]

            def read_uri(uri):
                if meta.encoding == None:  # data type is ".raw"
                    return get_first(
                        soundfile.read(
                            file=uri,
                            samplerate=meta.sample_rate,
                            channels=meta.channels,
                            dtype=meta.dtype,
                        )
                    )
                else:  # sound file auto detect the
                    return get_first(soundfile.read(file=uri))

            return read_uri
