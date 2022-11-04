import sys
from pathlib import Path
import os

import setuptools

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by Forte.")

VERSION_VAR = "VERSION"
version = {}
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "forte/version.py")
) as fp:
    exec(fp.read(), version)
if VERSION_VAR not in version or not version[VERSION_VAR]:
    raise ValueError(
        f"Cannot find {VERSION_VAR} in forte/version.py. Please make sure that "
        f"{VERSION_VAR} is correctly defined and formatted in forte/version.py."
    )

setuptools.setup(
    name="forte",
    version=version[VERSION_VAR],
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
        "enum34==1.1.10;python_version<'3.4'",
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
            "transformers>=4.15.0, <=4.22.2",
            "nltk",
            "texar-pytorch>=0.1.4",
            "requests",
        ],
        "ir": ["texar-pytorch>=0.1.4", "tensorflow>=1.15.0"],
        "remote": ["fastapi>=0.65.2, <=0.75.2", "pydantic<=1.9.2", "uvicorn>=0.14.0", "requests"],
        "audio_ext": ["soundfile>=0.10.3"],
        "stave": ["stave>=0.0.1.dev12"],
        "models": [
            "torch>=1.1.0",
            "torchtext==0.4.0",
            "tqdm>=4.36.1",
            "texar-pytorch>=0.1.4",
            "tensorflow>=1.15.0",
            "pyannote.audio",
            "pytorch-transformers",
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
        "ocr_tutorial": ["Pillow", "requests", "pytesseract"]
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
