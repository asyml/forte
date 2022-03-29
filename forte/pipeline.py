# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for Pipeline module.
"""

import os
import itertools
import json
import logging
import sys
import uuid
from pathlib import Path
from time import time
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Union,
    Tuple,
    Deque,
    Set,
)

import yaml
from pydantic import BaseModel

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.exception import (
    ProcessExecutionException,
    ProcessFlowException,
    ValidationError,
)
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.base_reader import BaseReader
from forte.data.caster import Caster
from forte.data.ontology.code_generation_objects import EntryTree
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator
from forte.data.selector import Selector, DummySelector
from forte.evaluation.base.base_evaluator import Evaluator
from forte.pipeline_component import PipelineComponent
from forte.process_job import ProcessJob
from forte.process_manager import ProcessManager, ProcessJobStatus
from forte.processors.base import BaseProcessor
from forte.processors.base.batch_processor import BaseBatchProcessor
from forte.utils import create_class_with_kwargs, get_full_module_name
from forte.utils.utils_processor import record_types_and_attributes_check
from forte.version import FORTE_IR_VERSION

if sys.version_info < (3, 7):
    import importlib_resources as resources
else:
    from importlib import resources

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "serve"]


class ProcessBuffer:
    def __init__(self, pipeline: "Pipeline", data_iter: Iterator[PackType]):
        self.__data_iter: Iterator[PackType] = data_iter
        self.__data_exhausted = False
        self.__pipeline = pipeline
        self.__process_manager: ProcessManager = pipeline._proc_mgr

    def __iter__(self):
        return self

    def __next__(self) -> ProcessJob:
        if self.__process_manager.current_queue_index == -1:
            if self.__data_exhausted:
                # Both the buffer is empty and the data input is exhausted.
                raise StopIteration
            try:
                job_pack = next(self.__data_iter)
                job = ProcessJob(job_pack, False)

                if len(self.__pipeline.evaluator_indices) > 0:
                    gold_copy = job_pack.view()
                    self.__pipeline.add_gold_packs({job.id: gold_copy})

                self.__process_manager.add_to_queue(queue_index=0, job=job)
                self.__process_manager.current_queue_index = 0
                self.__process_manager.current_processor_index = 0
                return job
            except StopIteration:
                self.__data_exhausted = True
                job = ProcessJob(None, True)
                self.__process_manager.add_to_queue(queue_index=0, job=job)
                self.__process_manager.current_queue_index = 0
                self.__process_manager.current_processor_index = 0
                return job
        else:
            q_index = self.__process_manager.current_queue_index
            u_index = self.__process_manager.unprocessed_queue_indices[q_index]
            return self.__process_manager.current_queue[u_index]


class Pipeline(Generic[PackType]):
    # pylint: disable=too-many-public-methods
    r"""This controls the main inference flow of the system. A pipeline is
    consisted of a set of Components (readers and processors). The data flows
    in the pipeline as data packs, and each component will use or add
    information to the data packs.
    """

    def __init__(
        self,
        resource: Optional[Resources] = None,
        ontology_file: Optional[str] = None,
        enforce_consistency: bool = False,
        do_init_type_check: bool = False,
    ):
        r"""

        Args:
            resource: The :class:`Resources` object, which is a global registry used
                in the pipeline. Objects defined as :class:`Resources` will be
                passed on to the processors in the
                pipeline for initialization.
            ontology_file: The path to the input ontology specification file,
                which should be a json file, and it should have all the entries
                inside with no import as key.
            enforce_consistency: This boolean determines whether the
                pipeline will check the content expectations specified in each
                pipeline component. Each component will check whether the input
                pack contains the expected data
                via checking the meta-data, and throws a
                :class:`~forte.common.exception.EntryNotFoundError` if it
                fails. When this function is called with enforce is ``True``,
                all the pipeline components would check if the input datapack
                record matches
                with the expected types and attributes if function
                ``expected_types_and_attributes`` is implemented
                for the processor. For example, processor A requires entry type
                of :class:`~ft.onto.base_ontology.Sentence`, and processor B would
                produce this type in the output datapack, so ``record`` function
                of processor B writes the record of this type in the datapack
                and processor A implements ``expected_types_and_attributes`` to
                add this type. Then when the pipeline runs with
                `enforce_consistency=True`, processor A would check if this
                type exists in the record of the output of the
                previous pipeline component.
            do_init_type_check: Determine whether to check records types and
                attributes during pipeline initialization. Default to `False`.
                If this boolean is set to `True`, each component in the
                pipeline will be validated by comparing its
                ``expected_types_and_attributes`` with the accumulated
                ``records`` from all the downstream components.
        """
        self._reader: BaseReader
        self._reader_config: Optional[Config] = None

        # These variables defines the units in the pipeline, they should be
        # of the same length
        self._components: List[PipelineComponent] = []
        self._selectors: List[Selector] = []
        self._configs: List[Optional[Config]] = []
        self._selectors_configs: List[Optional[Config]] = []
        # corresponding to the new added parameter "ref_name", indicating a list of
        # reference names that are used to identify different components
        self._ref_names: Dict[str, Any] = {}
        # Maintain a set of the pipeline components to fast check whether
        # the component is already there.
        self.__component_set: Set[PipelineComponent] = set()

        # Will initialize at `initialize` because the processors length is
        # unknown.
        self._proc_mgr: ProcessManager = None  # type: ignore

        self.evaluator_indices: List[int] = []

        # needed for evaluator
        self._predict_to_gold: Dict[int, PackType] = {}

        if resource is None:
            self.resource = Resources()
        else:
            self.resource = resource

        if ontology_file is None:
            with resources.path(
                "forte.ontology_specs", "base_ontology.json"
            ) as data_path:
                ontology_file = str(data_path)

        if ontology_file is not None:
            with open(ontology_file, "r", encoding="ascii") as f:
                spec_dict = json.load(f)
                self.resource.update(onto_specs_path=ontology_file)
                self.resource.update(onto_specs_dict=spec_dict)

        # The flag indicating whether this pipeline is initialized.
        self._initialized: bool = False
        # The flag indicating whether we want to enforce type consistency
        #  between the processors.
        self._check_type_consistency: bool = False

        # Create one copy of the dummy selector to reduce class creation.
        self.__default_selector: Selector = DummySelector()
        self.__default_selector_config: Config = Config({}, {})

        # needed for time profiling of pipeline
        self._enable_profiling: bool = False
        self._profiler: List[float] = []

        self._check_type_consistency = enforce_consistency

        # Indicate whether do type checking during pipeline initialization
        self._do_init_type_check: bool = do_init_type_check

    def enforce_consistency(self, enforce: bool = True):
        r"""This function determines whether the pipeline will check
        the content expectations specified in each pipeline component. This
        function works with :meth:`~forte.pipeline.Pipeline.initialize` called
        after itself. Each component will check whether the input pack contains
        the expected data via checking the meta-data, and throws a
        :class:`~forte.common.exception.EntryNotFoundError` if the check
        fails. The example of implementation is mentioned in the docstrings of
        :meth:`~forte.pipeline.Pipeline.__init__`.

        Args:
            enforce: A boolean of whether to enable consistency checking
                for the pipeline or not.
        """
        self._check_type_consistency = enforce

    def init_from_config_path(self, config_path):
        r"""Read the configurations from the given path ``config_path``
        and build the pipeline with the config.

        Args:
            config_path: A string of the configuration path, which is
                is a YAML file that specify the structure and parameters of the
                pipeline.
        """
        with open(config_path, encoding="utf-8") as f:
            configs = yaml.safe_load(f)
            self.init_from_config(configs)

    def init_from_config(self, configs: Dict[str, Any]):
        r"""Initialized the pipeline (ontology and processors) from the
        given configurations.

        Args:
            configs: The configs used to initialize the pipeline. It should be
                a dictionary that contains `forte_ir_version`, ``components``
                and `states`. `forte_ir_version` is a string used to validate
                input format. ``components`` is a list of dictionary that
                contains `type` (the class of pipeline components),
                `configs` (the corresponding component's configs) and
                `selector`. `states` will be used to update the pipeline states
                based on the fields specified in `states.attribute` and
                `states.resource`.
        """
        # Validate IR version
        if configs.get("forte_ir_version") != FORTE_IR_VERSION:
            raise ProcessorConfigError(
                f"forte_ir_version={configs.get('forte_ir_version')} not "
                "supported. Please make sure the format of input IR complies "
                f"with forte_ir_version={FORTE_IR_VERSION}."
            )

        # Add components from IR
        is_first: bool = True
        for component_config in configs.get("components", []):
            component = create_class_with_kwargs(
                class_name=component_config["type"],
                class_args=component_config.get("kwargs", {}),
            )

            if is_first:
                if not isinstance(component, BaseReader):
                    raise ProcessorConfigError(
                        "The first component of a pipeline must be a reader."
                    )
                self.set_reader(component, component_config.get("configs", {}))
                is_first = False
            else:
                # Can be processor, caster, or evaluator
                selector_config = component_config.get("selector")
                self.add(
                    component=component,
                    config=component_config.get("configs", {}),
                    selector=selector_config
                    and create_class_with_kwargs(
                        class_name=selector_config["type"],
                        class_args=selector_config.get("kwargs", {}),
                    ),
                    selector_config=None
                    if selector_config is None
                    else selector_config.get("configs"),
                )

        # Set pipeline states and resources
        states_config: Dict[str, Dict] = configs.get("states", {})
        for attr, val in states_config.get("attribute", {}).items():
            setattr(self, attr, val)
        resource_config: Dict[str, Dict] = states_config.get("resource", {})
        if "onto_specs_dict" in resource_config:
            self.resource.update(
                onto_specs_dict=resource_config["onto_specs_dict"]
            )
        if "merged_entry_tree" in resource_config:
            self.resource.update(
                merged_entry_tree=EntryTree().fromdict(
                    resource_config["merged_entry_tree"]
                ),
            )

    def _dump_to_config(self):
        r"""Serialize the pipeline to an IR(intermediate representation).
        The returned IR can be passed to `init_from_config` to initialize
        a pipeline.

        Returns:
            dict: A dictionary storing IR.
        """

        def test_jsonable(test_dict: Dict, err_msg: str):
            r"""Check if a dictionary is JSON serializable"""
            try:
                json.dumps(test_dict)
                return test_dict
            except (TypeError, OverflowError) as e:
                raise ProcessorConfigError(err_msg) from e

        get_err_msg: Dict = {
            "reader": lambda reader: (
                "The reader of the pipeline cannot be JSON serialized. This is"
                " likely due to some parameters in the configuration of the "
                f"reader {get_full_module_name(reader)} cannot be serialized "
                "in JSON. To resolve this issue, you can consider implementing"
                " a JSON serialization for that parameter type or changing the"
                " parameters of this reader. Note that in order for the reader"
                " to be serialized in JSON, all the variables defined in both "
                "the default_configs and the configuration passed in during "
                "pipeline.set_reader() need to be JSON-serializable. You can "
                "find in the stack trace the type of the un-serializable "
                "parameter."
            ),
            "component": lambda component: (
                "One component of the pipeline cannot be JSON serialized. This"
                " is likely due to some parameters in the configuration of the"
                f" component {get_full_module_name(component)} cannot be "
                "serialized in JSON. To resolve this issue, you can consider "
                "implementing a JSON serialization for that parameter type or "
                "changing the parameters of the component. Note that in order "
                "for the component to be serialized in JSON, all the variables"
                " defined in both the default_configs and the configuration "
                "passed in during pipeline.add() need to be JSON-serializable."
                " You can find in the stack trace the type of the "
                "un-serializable parameter."
            ),
            "selector": lambda selector: (
                "A selector cannot be JSON serialized. This is likely due to "
                "some __init__ parameters for class "
                f"{get_full_module_name(selector)} cannot be serialized in "
                "JSON. To resolve this issue, you can consider implementing a "
                "JSON serialization for that parameter type or changing the "
                "signature of the __init__ function. You can find in the stack"
                " trace the type of the un-serializable parameter."
            ),
        }

        configs: Dict = {
            "forte_ir_version": FORTE_IR_VERSION,
            "components": [],
            "states": {},
        }

        # Serialize pipeline components
        configs["components"].append(
            {
                "type": get_full_module_name(self._reader),
                "configs": test_jsonable(
                    test_dict=self._reader_config.todict(),
                    err_msg=get_err_msg["reader"](self._reader),
                ),
            }
        )
        for component, config, selector, selector_config in zip(
            self.components,
            self.component_configs,
            self._selectors,
            self._selectors_configs,
        ):
            configs["components"].append(
                {
                    "type": get_full_module_name(component),
                    "configs": test_jsonable(
                        test_dict=config.todict(),
                        err_msg=get_err_msg["component"](component),
                    ),
                    "selector": {
                        "type": get_full_module_name(selector),
                        "configs": test_jsonable(
                            # pylint: disable=protected-access
                            test_dict=selector_config.todict(),
                            # pylint: enable=protected-access
                            err_msg=get_err_msg["selector"](selector),
                        ),
                    },
                }
            )

        # Serialize current states of pipeline
        configs["states"].update(
            {
                "attribute": {
                    attr: getattr(self, attr)
                    for attr in (
                        "_initialized",
                        "_enable_profiling",
                        "_check_type_consistency",
                        "_do_init_type_check",
                    )
                    if hasattr(self, attr)
                },
                "resource": {},
            }
        )
        if self.resource.contains("onto_specs_dict"):
            configs["states"]["resource"].update(
                {"onto_specs_dict": self.resource.get("onto_specs_dict")}
            )
        if self.resource.contains("merged_entry_tree"):
            configs["states"]["resource"].update(
                {
                    "merged_entry_tree": self.resource.get(
                        "merged_entry_tree"
                    ).todict()
                }
            )

        return configs

    def save(self, path: str):
        r"""Store the pipeline as an IR(intermediate representation) in yaml.
        The path can then be passed to
        :meth:`~forte.pipeline.Pipeline.init_from_config_path` to initialize
        a pipeline. Note that calling
        :meth:`~forte.pipeline.Pipeline.init_from_config` from a different
        python environment may not work for some self defined component classes
        because their module name is `__main__`.

        Args:
            path: The file path to save configurations.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._dump_to_config(), f)

    def export(self, name: Optional[str] = None) -> Optional[str]:
        r"""Exports pipeline to FORTE_EXPORT_PATH.

        FORTE_EXPORT_PATH is a directory where all serialized pipeline will be stored.
        Users can specify through environment variable FORTE_EXPORT_PATH.

        This method will have the following behaviors:

            - FORTE_EXPORT_PATH will be created if assigned but not found.

            - If name is not provided, a default name `pipeline` will be used
              and suffixed by UUID, to prevent overwriting
              (e.g. `pipeline-4ba29336-aa05-11ec-abec-309c23414763.yml`).

            - If name is provided, then no suffix will be appended.

            - The pipeline is saved by :meth:`~forte.pipeline.Pipeline.save`,
              which exports the pipeline by :meth:`~forte.pipeline.Pipeline._dump_to_config`
              and saves it to a YAML file.

        Args:
            name: Export name of the pipeline. Default is None.

        Returns:
            Optional[str]: Export path of pipeline config YAML.

        Raises:
            ValueError: if export name is already taken.
        """
        export_path: Optional[str] = os.environ.get("FORTE_EXPORT_PATH")
        if export_path:
            os.makedirs(export_path, exist_ok=True)

            if name:
                export_name = f"{name}.yml"
            else:
                suffix = str(uuid.uuid1())
                export_name = f"pipeline-{suffix}.yml"

            export_path = os.path.join(export_path, export_name)
            if Path(export_path).exists():
                raise ValueError(f"{export_path} already exists.")

            self.save(export_path)
        return export_path

    def _remote_service_app(
        self, service_name: str = "", input_format: str = "string"
    ):
        r"""Return a FastAPI app that can be used to serve the pipeline.

        Args:
            service_name: Assign a name to the pipeline service for validation.
                This will appear in the `service_name` field on default page
                and can be queried and validated against the expected service
                name set by user. Default to `''`.
            input_format: Specify format of the input for validation. It can be
                `"string"` or `"DataPack"`. This will appear in the
                `input_format` field on default page and can be queried and
                validated against the expected input format set by user.
                Default to `"string"`.

        Returns:
            FastAPI: A FastAPI app for remote service.

        Raises:
            ImportError: An error occurred importing `fastapi` module.
        """
        # TODO: Currently we only support the `process` function, but it can
        # be extended by adding new interfaces that wrap up any Pipeline
        # method. Refer to https://fastapi.tiangolo.com for more info.
        try:
            # pylint:disable=import-outside-toplevel
            from fastapi import FastAPI
        except ImportError as e:
            raise ImportError(
                "'fastapi' must be installed to get a service app of "
                "pipeline. You can run 'pip install forte[remote]' to "
                "install all the requirements needed to start a pipeline "
                "service."
            ) from e

        app = FastAPI()
        records: Optional[Dict[str, Set[str]]] = None

        class RequestBody(BaseModel):
            args: str = "[]"
            kwargs: str = "{}"

        # pylint: disable=unused-variable
        @app.get("/")
        def default_page():
            return {
                "status": "OK",
                "service_name": service_name,
                "input_format": input_format,
                "pipeline": self._dump_to_config(),
            }

        @app.get("/records")
        def get_records():
            nonlocal records
            if records is None:
                # Collect records of each pipeline component for validation
                records = {}
                for component in [self._reader] + self.components:
                    if hasattr(component, "record"):
                        component.record(records)
            return {"status": "OK", "records": records}

        @app.get("/expectation")
        def get_expectation():
            expectation: Dict[str, Set[str]] = {}
            if len(self.components) > 0 and hasattr(
                self.components[0], "expected_types_and_attributes"
            ):
                expectation = self.components[0].expected_types_and_attributes()
            return {"status": "OK", "expectation": expectation}

        @app.post("/process")
        def run_pipeline(body: RequestBody):
            args = json.loads(body.args)
            kwargs = json.loads(body.kwargs)
            result = self.process(*args, **kwargs)
            return {"status": "OK", "result": result.to_string()}

        # pylint: enable=unused-variable

        return app

    def serve(
        self,
        host: str = "localhost",
        port: int = 8008,
        service_name: str = "",
        input_format: str = "string",
    ):
        r"""Start a service of the current pipeline at a specified host
        and port.

        Args:
            host: Port number of pipeline service.
            port: Host name of pipeline service.
            service_name: Assign a name to the pipeline service for validation.
                This will appear in the `service_name` field on default page
                and can be queried and validated against the expected service
                name set by user. Default to `''`.
            input_format: Specify format of the input for validation. It can be
                `"string"` or `"DataPack"`. This will appear in the
                `input_format` field on default page and can be queried and
                validated against the expected input format set by user.
                Default to `"string"`.

        Raises:
            ImportError: An error occurred importing `uvicorn` module.
        """
        if "uvicorn" not in sys.modules:
            try:
                import uvicorn  # pylint: disable=import-outside-toplevel
            except ImportError as e:
                raise ImportError(
                    "'uvicorn' must be installed to start a service of "
                    "pipeline. You can run 'pip install forte[remote]' to "
                    "install all the requirements needed to start a pipeline "
                    "service."
                ) from e

        self.initialize()
        uvicorn.run(
            self._remote_service_app(
                service_name=service_name, input_format=input_format
            ),
            host=host,
            port=port,
            log_level="info",
        )

    def set_profiling(self, enable_profiling: bool = True):
        r"""Set profiling option.

        Args:
            enable_profiling: A boolean of whether to enable profiling
                for the pipeline or not (the default is True).
        """
        self._enable_profiling = enable_profiling

    def initialize(self) -> "Pipeline":
        """
        This function should be called before the pipeline can be used to
        process the actual data. This function will call the `initialize` of
        all the components inside this pipeline.

        Returns:
            None
        """
        # create EntryTree type object merged_entry_tree to store the parsed
        # entry tree from ontology specification file passed in as part of
        # resource and add the result to resource with key of merged_entry_tree.
        merged_entry_tree = EntryTree()
        if self.resource.get("onto_specs_path"):
            OntologyCodeGenerator().parse_schema_for_no_import_onto_specs_file(
                ontology_path=self.resource.get("onto_specs_path"),
                ontology_dict=self.resource.get("onto_specs_dict"),
                merged_entry_tree=merged_entry_tree,
            )
            self.resource.update(merged_entry_tree=merged_entry_tree)

        # The process manager need to be assigned first.
        self._proc_mgr = ProcessManager(len(self._components))

        if self._initialized:
            # The pipeline has already been initialized, so we are doing
            # re-initialization here.
            logging.info("Re-initializing the Pipeline.")

        # Reset the flags of the components before initializing them.
        self._reader.reset_flags()
        for c in self._components:
            c.reset_flags()

        # Handle the reader.
        if not self._reader.is_initialized:
            self._reader.initialize(self.resource, self._reader_config)
        else:
            logging.info(
                "The reader [%s] has already initialized, "
                "will skip its initialization.",
                self._reader.name,
            )

        if self._check_type_consistency:
            self.reader.enforce_consistency(enforce=True)
        else:
            self.reader.enforce_consistency(enforce=False)

        # Handle other components and their selectors.
        self.initialize_components()
        self.initialize_selectors()
        self._initialized = True

        # Create profiler
        if self._enable_profiling:
            self.reader.set_profiling(True)
            self._profiler = [0.0] * len(self.components)

        # Check record types and attributes of each pipeline component
        if self._do_init_type_check:
            current_records: Dict[str, Set[str]] = {}
            self._reader.record(current_records)
            for component in self.components:
                if hasattr(component, "expected_types_and_attributes"):
                    record_types_and_attributes_check(
                        component.expected_types_and_attributes(),  # type: ignore
                        current_records,
                    )
                if hasattr(component, "record"):
                    component.record(current_records)  # type: ignore

        return self

    def initialize_components(self):
        """
        This function will initialize all the components in this pipeline,
        except the reader. The components are initialized in a FIFO manner
        based on the order of insertion,

        During initialization, the component will be configured based on its
        corresponding configuration. However, if the component is already
        initialized (for example, being initialized manually or used twice
        in the same pipeline), the new configuration will be ignored.

        The pipeline will check for type dependencies between the components
        inside this pipeline, see
        :func:`~forte.pipeline_component.PipelineComponent.enforce_consistency`
        for more details.

        """
        for component, config in zip(self.components, self.component_configs):
            try:
                if not component.is_initialized:
                    component.initialize(self.resource, config)
                else:
                    logging.info(
                        "The component [%s] has already initialized, "
                        "will skip its initialization.",
                        component.name,
                    )
            except ProcessorConfigError as e:
                logging.error(
                    "Exception occur when initializing " "processor %s",
                    component.name,
                )
                raise e

            component.enforce_consistency(enforce=self._check_type_consistency)

    def initialize_selectors(self):
        """
        This function will reset the states of selectors
        """
        for selector, config in zip(self._selectors, self._selectors_configs):
            try:
                selector.initialize(config)
            except ValueError as e:
                logging.error("Exception occur when initializing selectors")
                raise e

    def set_reader(
        self,
        reader: BaseReader,
        config: Optional[Union[Config, Dict[str, Any]]] = None,
    ) -> "Pipeline":
        """
        Set the reader of the pipeline. A reader is the entry point of
        this pipeline, data flown into the reader will be converted to the
        data pack format, and being passed onto the other components for
        processing.

        Args:
            reader: The reader to be used of the pipeline
            config: The custom configuration to be passed to the reader. If
              the config is not provided, the default config defined by the
              reader class will be used.

        Returns:
            The pipeline itself, which allows you to directly chain other
            pipeline construction code afterwards, i.e., you can do:

            .. code-block:: python

                Pipeline().set_reader(your_reader()).add(your_processor())

        """
        self._reader = reader
        self._reader_config = reader.make_configs(config)
        return self

    @property
    def reader(self) -> BaseReader:
        return self._reader

    @property
    def components(self) -> List[PipelineComponent]:
        """
        Return all the components in this pipeline, except the reader.

        Returns: A list containing the components.

        """
        return self._components

    @property
    def ref_names(self) -> Dict[str, int]:
        """
        Return all the reference names in this pipeline, except the reader.

        Returns: A dictionary containing the reference names.

        """
        return self._ref_names

    @property
    def component_configs(self) -> List[Optional[Config]]:
        """
        Return the configs related to the components, except the reader.

        Returns: A list containing the components configs.

        """
        return self._configs

    def add(
        self,
        component: PipelineComponent,
        config: Optional[Union[Config, Dict[str, Any]]] = None,
        selector: Optional[Selector] = None,
        selector_config: Optional[Union[Config, Dict[str, Any]]] = None,
        ref_name: Optional[str] = None,
    ) -> "Pipeline":
        """
        Adds a pipeline component to the pipeline. The pipeline components
        will form a chain based on the insertion order. The customized
        `config` and `selector` (:class:`~forte.data.selector.Selector`)
        will be associated with this particular component. If the `config`
        or the `selector` is not provided, the default ones will be used.

        Here, note that the same component instance can be added multiple
        times to the pipeline. In such cases, the instance will only be
        setup at the first insertion (i.e. its `initialize` function will
        only be called once). The subsequent insertion of the same component
        instance will not change the behavior nor the states of the instance.
        Thus, a different `config` cannot be provided (should be `None`) when
        added the second time, otherwise a `ProcessorConfigError` will be
        thrown. In the case where one want to them to behave differently, a
        different instance should be used.

        Args:
            component (PipelineComponent): The component to be inserted next
              to the pipeline.
            config (Union[Config, Dict[str, Any]): The custom configuration
              to be used for the added component. Default None, which means
              the `default_configs()` of the component will be used.
            selector (Selector): The selector used to pick the corresponding
              data pack to be consumed by the component. Default None, which
              means the whole pack will be used.

        Returns:
            The pipeline itself, which enables one to chain the creation of
            the pipeline, i.e., you can do:

            .. code-block:: python

                Pipeline().set_reader(your_reader()).add(
                    your_processor()).add(anther_processor())
        """
        if isinstance(component, BaseReader):
            raise ProcessFlowException("Reader need to be set via set_reader()")

        if isinstance(component, Evaluator):
            # This will ask the job to keep a copy of the gold standard.
            self.evaluator_indices.append(len(self.components))

        if ref_name is not None:
            if ref_name in self._ref_names:
                raise ValidationError(
                    f"This reference name {ref_name} already exists, please specify a new one"
                )
            else:
                self._ref_names[ref_name] = len(self.components)

        if component not in self.__component_set:
            # The case where the component is not found.
            self._components.append(component)
            self.__component_set.add(component)
            self.component_configs.append(component.make_configs(config))
        else:
            if config is None:
                self._components.append(component)
                # We insert a `None` value here just to make the config list
                # to match the component list, but this config should not be
                # used.
                self.component_configs.append(None)
            else:
                raise ProcessorConfigError(
                    f"The same instance of a component named {component.name} "
                    f" has already been added to"
                    f" the pipeline, we do not accept a different configuration"
                    f" for it. If you would like to use a differently"
                    f" configured component, please create another instance."
                    f" If you intend to re-use the component instance, please"
                    f" do not provide the `config` (or provide a `None`)."
                )

        if selector is None:
            self._selectors.append(self.__default_selector)
            self._selectors_configs.append(self.__default_selector_config)
        else:
            self._selectors.append(selector)
            self._selectors_configs.append(
                selector.make_configs(selector_config)
            )

        return self

    def add_gold_packs(self, pack):
        r"""Add gold packs to a internal dictionary used for evaluation.
        This dictionary is used by the evaluator while calling
        `consume_next(...)`

        Args:
            pack (Dict): A key, value pair containing job.id -> gold_pack
                mapping
        """
        self._predict_to_gold.update(pack)

    def process(self, *args, **kwargs) -> PackType:
        r"""Alias for :meth:`process_one`.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        return self.process_one(*args, **kwargs)

    def run(self, *args, **kwargs):
        r"""Run the whole pipeline and ignore all returned DataPack. This is
        mostly used when you need to run the pipeline and do not require the
        output but rely on the side-effect. For example, if the pipeline
        writes some data to disk.

        Calling this function will automatically call the :meth:`initialize`
        at the beginning, and call the :meth:`finish` at the end.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        self.initialize()
        for _ in self.process_dataset(*args, **kwargs):
            # Process the whole dataset ignoring the return values.
            # This essentially expect the processors have side effects.
            pass
        self.finish()

    def process_one(self, *args, **kwargs) -> PackType:
        r"""Process one single data pack. This is done by only reading and
        processing the first pack in the reader.

        Args:
            kwargs: the information needed to load the data. For example, if
                :attr:`_reader` is
                :class:`~forte.data.readers.string_reader.StringReader`, this
                should contain a
                single piece of text in the form of a string variable. If
                :attr:`_reader` is a file reader, this can point to the file
                path.
        """
        if not self._initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline"
            )

        first_pack = []

        for p in self._reader.iter(*args, **kwargs):
            first_pack.append(p)
            break

        if len(first_pack) == 1:
            results = list(self._process_packs(iter(first_pack)))
            return results[0]
        else:
            raise ValueError("Input data source contains no packs.")

    def process_dataset(self, *args, **kwargs) -> Iterator[PackType]:
        r"""Process the documents in the data source(s) and return an
        iterator or list of DataPacks. The arguments are directly passed
        to the reader to take data from the source.
        """
        if not self._initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline"
            )

        data_iter = self._reader.iter(*args, **kwargs)
        return self._process_packs(data_iter)

    def finish(self):
        """
        Call the finish method of all pipeline component. This need to be called
        explicitly to release all resources.
        """

        # Report time profiling of readers and processors
        if self._enable_profiling:
            out_header: str = "Pipeline Time Profile\n"
            out_reader: str = (
                f"- Reader: {self.reader.component_name}, "
                + f"{self.reader.time_profile} s\n"
            )
            out_processor: str = "\n".join(
                [
                    f"- Component [{i}]: {self.components[i].name}, {t} s"
                    for i, t in enumerate(self._profiler)
                ]
            )
            logger.info("%s%s%s", out_header, out_reader, out_processor)

        self.reader.finish(self.resource)
        for p in self.components:
            p.finish(self.resource)
        self._initialized = False

    def __update_stream_job_status(self):
        q_index = self._proc_mgr.current_queue_index
        u_index = self._proc_mgr.unprocessed_queue_indices[q_index]
        current_queue = self._proc_mgr.current_queue

        for job_i in itertools.islice(current_queue, 0, u_index + 1):
            if job_i.status == ProcessJobStatus.UNPROCESSED:
                job_i.set_status(ProcessJobStatus.PROCESSED)

    def __update_batch_job_status(self, component: BaseBatchProcessor):
        # update the status of the jobs. The jobs which were removed from
        # data_pack_pool will have status "PROCESSED" else they are "QUEUED"
        q_index = self._proc_mgr.current_queue_index
        u_index = self._proc_mgr.unprocessed_queue_indices[q_index]
        current_queue = self._proc_mgr.current_queue

        data_pool_length = len(component.batcher.data_pack_pool)

        for i, job_i in enumerate(
            itertools.islice(current_queue, 0, u_index + 1)
        ):
            if i <= u_index - data_pool_length:
                job_i.set_status(ProcessJobStatus.PROCESSED)
            else:
                job_i.set_status(ProcessJobStatus.QUEUED)

    def __flush_batch_job_status(self):
        current_queue = self._proc_mgr.current_queue
        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)

    def _process_with_component(
        self,
        selector: Selector,
        component: PipelineComponent,
        raw_job: ProcessJob,
    ):
        for pack in selector.select(raw_job.pack):
            # First, perform the component action on the pack
            try:
                if isinstance(component, Caster):
                    # Replacing the job pack with the casted version.
                    raw_job.alter_pack(component.cast(pack))
                elif isinstance(component, BaseBatchProcessor):
                    pack.set_control_component(component.name)
                    component.process(pack)
                elif isinstance(component, Evaluator):
                    pack.set_control_component(component.name)
                    component.consume_next(
                        pack, self._predict_to_gold[raw_job.id]
                    )
                elif isinstance(component, BaseProcessor):
                    # Should be BasePackProcessor:
                    # All other processor are considered to be
                    # streaming processor like this.
                    pack.set_control_component(component.name)
                    component.process(pack)
                # After the component action, make sure the entry is
                # added into the index.
                pack.add_all_remaining_entries()
            except ValueError as e:
                logger.error(e, exc_info=True)
                raise ProcessExecutionException(
                    f"Exception occurred when running " f"{component.name}"
                ) from e

    def _process_packs(
        self, data_iter: Iterator[PackType]
    ) -> Iterator[PackType]:
        r"""Process the packs received from the reader by the running through
        the pipeline.

        Args:
             data_iter (iterator): Iterator yielding jobs that contain packs

        Returns:
            Yields packs that are processed by the pipeline.
        """

        # pylint: disable=line-too-long

        # Here is the logic for the execution of the pipeline.

        # The basic idea is to yield a pack as soon as it gets processed by all
        # the processors instead of waiting for later jobs to get processed.

        # 1) A job can be in three status
        #  - UNPROCESSED
        #  - QUEUED
        #  - PROCESSED
        #
        # 2) Each processor maintains a queue to hold the jobs to be executed
        # next.
        #
        # 3) In case of a BatchProcessor, a job enters into QUEUED status if the
        # batch is not full according to the batcher of that processor.
        # In that case, the pipeline requests for additional jobs from the
        # reader and starts the execution loop from the beginning.
        #
        # 4) At any point, while moving to the next processor, the pipeline
        # ensures that all jobs are either in QUEUED or PROCESSED status. If
        # they are PROCESSED, they will be moved to the next queue. This design
        # ensures that at any point, while processing the job at processor `i`,
        # all the jobs in the previous queues are in QUEUED status. So whenever
        # a new job is needed, the pipeline can directly request it from the
        # reader instead of looking at previous queues for UNPROCESSED jobs.
        #
        # 5) When a processor receives a poison pack, it flushes all the
        # remaining batches in its memory (this actually has no effect in
        # PackProcessors) and moves the jobs including the poison pack to the
        # next queue. If there is no next processor, the packs are yield.
        #
        # 6) The loop terminates when the last queue contains only a poison pack
        #
        # Here is the sample pipeline and its execution
        #
        # Assume 1 pack corresponds to a batch of size 1
        #
        # After 1st step (iteration), reading from the reader,
        #
        #            batch_size = 2                               batch_size = 2
        #  Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |___________|
        #          |___________|
        #          |___________|
        #          |___________|
        #          |_J1:QUEUED_|
        #
        # B1 needs another pack to process job J1
        #
        # After 2nd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |___________|       |_______________|
        #          |___________|       |_______________|
        #          |___________|       |_______________|
        #          |___________|       |_J2:UNPROCESSED_|
        #          |___________|       |_J1:UNPROCESSED_|
        #
        # B1 processes both the packs, the jobs are moved to the next queue.
        #
        # After 3rd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_J2:UNPROCESSED_|     |_J1:UNPROCESSED_|
        #
        # P1 processes the first job. However, there exists one UNPROCESSED job
        # J2 in the queue. Pipeline first processes this job before moving to the
        # next processor
        #
        # After 4th step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_J2:UNPROCESSED_|
        #        |___________|       |_______________|     |_J1:UNPROCESSED_|
        #
        #
        # After 5th step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|    --> Yield J1.pack and J2.pack
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|

        if not self._initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline"
            )

        buffer = ProcessBuffer(self, data_iter)

        if len(self.components) == 0:
            yield from data_iter
            # Write return here instead of using if..else to reduce indent.
            return

        while not self._proc_mgr.exhausted():
            # Take the raw job from the buffer, the job status now should
            # be UNPROCESSED.
            raw_job: ProcessJob = next(buffer)

            component_index = self._proc_mgr.current_processor_index
            component = self.components[component_index]
            selector: Selector = self._selectors[component_index]
            current_queue_index = self._proc_mgr.current_queue_index
            current_queue: Deque[ProcessJob] = self._proc_mgr.current_queue
            pipeline_length = self._proc_mgr.pipeline_length
            unprocessed_queue_indices = self._proc_mgr.unprocessed_queue_indices
            processed_queue_indices = self._proc_mgr.processed_queue_indices
            next_queue_index = current_queue_index + 1
            should_yield = next_queue_index >= pipeline_length

            if not raw_job.is_poison:
                # Start timer
                if self._enable_profiling:
                    start_time: float = time()

                self._process_with_component(selector, component, raw_job)

                # Stop timer and add to time profiler
                if self._enable_profiling:
                    self._profiler[component_index] += time() - start_time

                # Then, based on component type, handle the queue.
                if isinstance(component, BaseBatchProcessor):
                    self.__update_batch_job_status(component)
                    index = unprocessed_queue_indices[current_queue_index]

                    # Check status of all the jobs up to "index".
                    for i, job_i in enumerate(
                        itertools.islice(current_queue, 0, index + 1)
                    ):
                        if job_i.status == ProcessJobStatus.PROCESSED:
                            processed_queue_indices[current_queue_index] = i

                    # There are UNPROCESSED jobs in the queue.
                    if index < len(current_queue) - 1:
                        unprocessed_queue_indices[current_queue_index] += 1
                    elif processed_queue_indices[current_queue_index] == -1:
                        # Fetch more data from the reader to process the
                        # first job.
                        unprocessed_queue_indices[current_queue_index] = len(
                            current_queue
                        )
                        self._proc_mgr.current_processor_index = 0
                        self._proc_mgr.current_queue_index = -1
                    else:
                        processed_queue_index = processed_queue_indices[
                            current_queue_index
                        ]
                        # Move or yield the pack.
                        c_queue = list(current_queue)
                        for job_i in c_queue[: processed_queue_index + 1]:
                            if job_i.status == ProcessJobStatus.PROCESSED:
                                if should_yield:
                                    if job_i.id in self._predict_to_gold:
                                        self._predict_to_gold.pop(job_i.id)
                                    # TODO: I don't know why these are
                                    #  marked as incompatible type by mypy.
                                    #  the same happens 3 times on every yield.
                                    #  It is observed that the pack returned
                                    #  from the `ProcessJob` is considered to
                                    #  be different from `PackType`.
                                    yield job_i.pack  # type: ignore
                                else:
                                    self._proc_mgr.add_to_queue(
                                        queue_index=next_queue_index, job=job_i
                                    )
                            else:
                                raise ProcessFlowException(
                                    f"The job status should be "
                                    f"{ProcessJobStatus.PROCESSED} "
                                    f"at this point."
                                )
                            current_queue.popleft()

                        # Set the UNPROCESSED and PROCESSED indices.
                        unprocessed_queue_indices[current_queue_index] = len(
                            current_queue
                        )

                        processed_queue_indices[current_queue_index] = -1

                        if should_yield:
                            self._proc_mgr.current_processor_index = 0
                            self._proc_mgr.current_queue_index = -1
                        else:
                            self._proc_mgr.current_processor_index = (
                                next_queue_index
                            )
                            self._proc_mgr.current_queue_index = (
                                next_queue_index
                            )
                # Besides Batch Processors, the other component type only
                # deal with one pack at a time, these include: PackProcessor
                # Evaluator, Caster.
                # - Move them to the next queue
                else:
                    self.__update_stream_job_status()
                    index = unprocessed_queue_indices[current_queue_index]

                    # there are UNPROCESSED jobs in the queue
                    if index < len(current_queue) - 1:
                        unprocessed_queue_indices[current_queue_index] += 1
                    else:
                        # current_queue is modified in this array
                        for job_i in list(current_queue):
                            if job_i.status == ProcessJobStatus.PROCESSED:
                                if should_yield:
                                    if job_i.id in self._predict_to_gold:
                                        self._predict_to_gold.pop(job_i.id)
                                    yield job_i.pack  # type: ignore
                                else:
                                    self._proc_mgr.add_to_queue(
                                        queue_index=next_queue_index, job=job_i
                                    )
                                current_queue.popleft()
                            else:
                                raise ProcessFlowException(
                                    f"The job status should be "
                                    f"{ProcessJobStatus.PROCESSED} "
                                    f"at this point."
                                )

                        # set the UNPROCESSED index
                        # we do not use "processed_queue_indices" as the
                        # jobs get PROCESSED whenever they are passed
                        # into a PackProcessor
                        unprocessed_queue_indices[current_queue_index] = len(
                            current_queue
                        )

                        # update the current queue and processor only
                        # when all the jobs are processed in the current
                        # queue
                        if should_yield:
                            self._proc_mgr.current_processor_index = 0
                            self._proc_mgr.current_queue_index = -1

                        else:
                            self._proc_mgr.current_processor_index = (
                                next_queue_index
                            )
                            self._proc_mgr.current_queue_index = (
                                next_queue_index
                            )
            else:
                component.flush()
                self.__flush_batch_job_status()

                # current queue is modified in the loop
                for job in list(current_queue):
                    if (
                        job.status != ProcessJobStatus.PROCESSED
                        and not job.is_poison
                    ):
                        raise ValueError(
                            "Job is neither PROCESSED nor is "
                            "a poison. Something went wrong "
                            "during execution."
                        )

                    if not job.is_poison and should_yield:
                        if job.id in self._predict_to_gold:
                            self._predict_to_gold.pop(job.id)
                        yield job.pack  # type: ignore

                    elif not should_yield:
                        self._proc_mgr.add_to_queue(
                            queue_index=next_queue_index, job=job
                        )

                    if not job.is_poison:
                        current_queue.popleft()

                if not should_yield:
                    # set next processor and queue as current
                    self._proc_mgr.current_processor_index = next_queue_index
                    self._proc_mgr.current_queue_index = next_queue_index

        self._proc_mgr.reset()

    def evaluate(self) -> Iterator[Tuple[str, Any]]:
        """
        Call the evaluators in the pipeline to collect their results.

        Returns:
            Iterator of the evaluator results. Each element is a tuple, where
            the first one is the name of the evaluator, and the second one
            is the output of the evaluator (see
            :func:`~forte.evaluation.base.base_evaluator.Evaluator.get_result`).
        """
        for i in self.evaluator_indices:
            p = self.components[i]
            assert isinstance(p, Evaluator)
            yield p.name, p.get_result()

    def get_component(self, ref_name: str) -> PipelineComponent[Any]:
        """
        Call the evaluator in the pipeline by the reference name to get a component.

        Args:
            ref_name(str): the reference name of a component
        """
        p = self.components[self.ref_names[ref_name]]
        return p


def serve(
    pl_config_path: str,
    host: str = "localhost",
    port: int = 8008,
    service_name: str = "",
    input_format: str = "string",
):
    r"""Start a remote service of a pipeline initialized from a YAML config at
    a specified host and port.

    Args:
        pl_config_path: A string of the configuration path, which is
            is a YAML file that specify the structure and parameters of the
            pipeline.
        host: Port number of pipeline service.
        port: Host name of pipeline service.
        service_name: Assign a name to the pipeline service for validation.
                This will appear in the `service_name` field on default page
                and can be queried and validated against the expected service
                name set by user. Default to `''`.
        input_format: Specify format of the input for validation. It can be
            `"string"` or `"DataPack"`. This will appear in the
            `input_format` field on default page and can be queried and
            validated against the expected input format set by user.
            Default to `"string"`.
    """
    pipeline: Pipeline = Pipeline()
    pipeline.init_from_config_path(pl_config_path)
    pipeline.serve(
        host=host,
        port=port,
        service_name=service_name,
        input_format=input_format,
    )
