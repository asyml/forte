Reader
=======

A pipeline component that reads data from a file path into a data iterator.



Examples
---------

We have an working MT translation pipeline example here https://github.com/asyml/forte/blob/master/docs/notebook_tutorial/wrap_MT_inference_pipeline.ipynb

This example uses :class:`PlainTextReader` to read txt file.

.. code-block:: python

    class PlainTextReader(PackReader):
        r""":class:`PlainTextReader` is designed to read in plain text dataset."""

        def _collect(self, text_directory) -> Iterator[Any]:  # type: ignore
            r"""Should be called with param ``text_directory`` which is a path to a
            folder containing txt files.

            Args:
                text_directory: text directory containing the files.

            Returns: Iterator over paths to .txt files
            """
            # dataset_path_iterator is a function that return all file paths
            # with the given file extensions
            # under the given directories
            return dataset_path_iterator(text_directory, self.configs.file_ext)

        def _cache_key_function(self, text_file: str) -> str:
            # it returns text basename as a cache key
            return os.path.basename(text_file)

        # pylint: disable=unused-argument
        def text_replace_operation(self, text: str):
            return []

        def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
            pack = DataPack()

            with open(file_path, "r", encoding="utf8", errors="ignore") as file:
                text = file.read()
            # set DataPack.text field to text after a list of replacing operation
            # in this reader, we don't have the list is empty so we don't have replace operations
            pack.set_text(text, replace_func=self.text_replace_operation)

            # Wrap data into ontology classes
            Document(pack, 0, len(pack.text))

            # set pack_name to file_path
            pack.pack_name = file_path
            yield pack

        @classmethod
        def default_configs(cls):
            return {"file_ext": ".txt"}

        def record(self, record_meta: Dict[str, Set[str]]):
            r"""Method to add output type record of `PlainTextReader` which is
            `ft.onto.base_ontology.Document` with an empty set
            to :attr:`forte.data.data_pack.Meta.record`.

            Args:
                record_meta: the field in the datapack for type record that need to
                    fill in for consistency checking.
            """
            record_meta["ft.onto.base_ontology.Document"] = set()


Usage
------





Functions
------------------

Based on the usage listed above, we need to customize functions below.
generic class method
- `_cache_key_function`.
    * key of a basic unit of the raw data.
    * Example from from classification reader:

    .. code-block:: python

        def _cache_key_function(self, line_info: Tuple[int, List[str]]) -> str:
            return str(line_info[0])

- `_parse_pack`
    * load a basic unit of raw data into data pack. It's also a process of structuralizing the data --- wrap data into ontology classes and data pack data fields.
    * Example from PlainTextReader:
    .. code-block:: python

        def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
            pack = DataPack()
            with open(file_path, "r", encoding="utf8", errors="ignore") as file:
                text = file.read()
            # writing into data pack data fields
            pack.set_text(text, replace_func=self.text_replace_operation)
            pack.pack_name = file_path
            # Wrap data into ontology classes
            Document(pack, 0, len(pack.text))

            yield pack

- `_collect`
    * read data from the path and iterate the data in raw format and return the basic unit (for example, a line of text while reading table-like data).
    * Example from ClassificationDatasetReader: https://github.com/asyml/forte/blob/4bb8fa5bd0be960426be223f0d295b9786c49b0a/forte/data/readers/classification_reader.py#L26
        - it uses csv reader to read csv table-like data
        - it skips line if `configs.skip_k_starting_lines` is set to be positive integer
        - it returns a iterator that yields a line id and a table row for each iteration.

    .. code-block:: python

        def _collect(  # type: ignore
            self, csv_file: str
        ) -> Iterator[Tuple[int, List[str]]]:
            with open(csv_file, encoding="utf-8") as f:
                # reading data
                data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
                if self.configs.skip_k_starting_lines > 0:
                    for _ in range(self.configs.skip_k_starting_lines):
                        next(data)
                # yield data as an interator
                for line_id, line in enumerate(data):
                    yield line_id, line





Reader Class Hierarchy
------------------------

Here we provide a simplified class hierarchy for :class:`PlainTextReader` to show the relations of readers which are subclasses of :class:`PipelineComponent`.

* `PipelineComponent`: As the hierarchy suggests, readers are subclasses of ~PipelineComponent
    * `BaseReader`
        - `PackReader`: a reader that reads data into :class:`DataPack`
            * `PlainTextReader`
        - `MultiPackReader`: a reader that reads data into :class:`MultiPack`.
        - ...
    * ...

* we have plenty of written reader available to use. If you don't find one suitable in your case, you can refer to this documentation and tutorials to create a new reader.
