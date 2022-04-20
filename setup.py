import sys
from pathlib import Path

import setuptools

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by Forte.")

setuptools.setup(
    name="forte",
    version="0.1.2",
    url="https://github.com/asyml/forte",
    description="Forte is extensible framework for building composable and "
    "modularized NLP workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=setuptools.find_namespace_packages(
        include=["ft.*", "ftx.*", "forte"]
    ),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "sortedcontainers>=2.1.0",
        "numpy>=1.16.6",
        "jsonpickle>=1.4",
        "pyyaml>=5.4",
        "smart-open>=1.8.4",
        "typed_astunparse>=2.1.4",
        "funcsigs>=1.0.2",
        "typed_ast>=1.5.0",
        "jsonschema>=3.0.2",
        'typing>=3.7.4;python_version<"3.5"',
        "typing-inspect>=0.6.0",
        'dataclasses~=0.7;python_version<"3.7"',
        'importlib-resources>=5.1.4;python_version<"3.7"',
        "asyml-utilities",
    ],
    extras_require={
        "data_aug": [
            "transformers>=4.15.0",
            "nltk",
            "texar-pytorch>=0.1.4",
        ],
        "ir": ["texar-pytorch>=0.1.4", "tensorflow>=1.15.0"],
        "remote": [
            "fastapi>=0.65.2",
            "uvicorn>=0.14.0",
        ],
        "audio_ext": ["soundfile>=0.10.3"],
        "stave": ["stave>=0.0.1.dev12"],
        "models": [
            "torch>=1.1.0",
            "torchtext==0.4.0",
            "tqdm>=4.36.1",
            "texar-pytorch>=0.1.4",
            "tensorflow>=1.15.0",
        ],
        "test": [
            "ddt",
            "testfixtures",
            "testbook",
            "termcolor",
            "transformers>=4.15.0",
            "nltk",
        ],
        "wikipedia": ["rdflib==4.2.2"],
        # transformers 4.10.0 will break the translation model we used here
        "nlp": ["texar-pytorch>=0.1.4"],
        "extractor": ["texar-pytorch>=0.1.4"],
    },
    entry_points={
        "console_scripts": [
            "generate_ontology = forte.command_line.generate_ontology.__main__:main"
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
