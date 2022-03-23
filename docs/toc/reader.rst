Reader
=======

A pipeline component that reads data from a data source into a data iterator.


Related Readings
------------------

#. `Reader API <../code/data.html#readers>`_


Functions
------------------

Based on the usage listed above, we need to customize functions below.
generic class method

- ``set_up()``: check correctness of configuration and initialize reader variables.
- ``initialize()``: Pipeline will call it at the start of processing. The reader
  will be initialized with ``configs``, and register global resources into
  ``resource``. The implementation should set up the states of the component.

    * ``default_configs`` is a class method that returns default configuration
      in a dictionary format. Parent reader class configuration will be merged
      or overwritten by child class.  For example, in the
      :class:`~forte.data.readers.plaintext_reader.PlainTextReader`,
      the inheritance chain is :class:`~forte.data.base_reader.BaseReader`
      ->
      :class:`~forte.data.base_reader.PackReader`
      ->
      :class:`~forte.data.readers.plaintext_reader.PlainTextReader`.
      :meth:`forte.data.base_reader.BaseReader.default_configs` contains
      ``"zip_pack"`` and ``"serialize_method"``.
      :meth:`forte.data.readers.plaintext_reader.PlainTextReader.default_configs`
      contains
      ``"file_ext"``. Therefore, the merged configuration contains
      ``zip_pack``,
      ``"serialize_method"`` and ``"file_ext"`` fields. Suppose we include
      ``"serialize_method"`` in
      :class:`~forte.data.readers.plaintext_reader.PlainTextReader`, we can
      overwrite the configuration in
      :class:`~forte.data.base_reader.BaseReader`.

        - ``default_configs`` usage example

            - To use an existing reader, User should check configurations from
              method ``default_configs()`` of the particular reader used to
              find
              out what configurations can be customized. For example, suppose
              after checking `reader API <../code/data.html#readers>`_ we
              decide to use :class:`~forte.data.base_reader.BaseReader`. Then
              we need to check the source of
              :meth:`forte.data.base_reader.BaseReader.default_configs()` and
              found that ``"zippack"`` is a boolean configuration and we can
              set it to ``True`` in our customized configuration when we don't
              want the default configuration. The default configuration will be
              overwritten when we initialize the reader with our customized
              configuration.

            - To implement a new reader, User should check the appropriate
              reader to inherit from. One consideration is whether User
              wants to read a data pack or a data pack batch for
              each reading iteration. If it's the
              :class:`~forte.data.data_pack.DataPack`,
              then User should inherit from
              :class:`~forte.data.base_reader.PackReader`.
              If it's :class:`~forte.data.multi_pack.MultiPack`,
              then User should inherit from
              :class:`~forte.data.base_reader.MultiPackReader`
              For example, in the
              :class:`~forte.data.readers.plaintext_reader.PlainTextReader`,
              it inherits
              from :class:`~forte.data.base_reader.PackReader` because it reads
              plain text into :class:`~forte.data.data_pack.DataPack`.
              Then User can
              consider adding new configuration field in ``default_configs()``
              based on the needs
              or overwrite the configuration field from its parent class.
              It's just a simple consideration to explain the process of
              choosing the right reader, there are many other readers
              with more features that User can inherit from. User can refer to
              `Readers API <../code/data.rst#Readers>`_ for more information.


        - ``default_configs`` code example

        .. code-block:: python

            @classmethod
            def default_configs(cls):
                return {"file_ext": ".txt"}


    - ``resource`` is for advanced developer. It's an shared object that stores data accessible by all ``PipelineComponent`` in the pipeline.



- ``_cache_key_function``.
    * it returns cache key of a unit of the data iterator returned by `_collect` such as a row id for a row in `csv` file reading.
    * Example from from `ClassificationDatasetReader <https://github.com/asyml/forte/blob/4bb8fa5bd0be960426be223f0d295b9786c49b0a/forte/data/readers/classification_reader.py#L140>`_ which use line id as cache key (it is `line_info[0]` in the line of code).:

- ``_parse_pack``
    * load a basic unit of raw data into data pack. It's also a process of structuralizing the data: wrap data into ontology classes and assign data to data fields.
    * Example from `PlainTextReader <https://github.com/asyml/forte/blob/0ca9602d3d287beb2521584f5fc50c2f5905cebc/forte/data/readers/plaintext_reader.py#L30>`_ which reads ``txt`` file.

    .. code-block:: python

        def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
            pack = DataPack()
            with open(file_path, "r", encoding="utf8", errors="ignore") as file:
                text = file.read()
            # writing into data pack data fields
            pack.set_text(text, replace_func=self.text_replace_operation)
            pack.pack_name = file_path
            # Wrap data into ontology classes
            # It also specifies the range of text for `Document`
            Document(pack, 0, len(pack.text))

            yield pack

- ``_collect``
    * read data from the data source and returns an iterator yields data (for example, a line of data while reading csv file).
    * Example from `ClassificationDatasetReader class   <https://github.com/asyml/forte/blob/4bb8fa5bd0be960426be223f0d295b9786c49b0a/forte/data/readers/classification_reader.py#L26>`_:
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




Examples
---------

We have an working MT translation pipeline example `here <https://github.com/asyml/forte/blob/master/docs/notebook_tutorial/wrap_MT_inference_pipeline.ipynb>`_

This example uses :class:`PlainTextReader` to read ``txt`` file.

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
            # dataset_path_iterator is a function that return all file paths with the given file extensions under the given directories
            return dataset_path_iterator(text_directory, self.configs.file_ext)

        def _cache_key_function(self, text_file: str) -> str:
            # it returns text basename as a cache key
            return os.path.basename(text_file)

        # pylint: disable=unused-argument
        def text_replace_operation(self, text: str):
            # this function returns a list of replacing operations
            # in this particular example, we don't have any replacing operation
            # so we return an empty list
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
