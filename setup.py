import sys
from pathlib import Path
import setuptools

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Forte.')

setuptools.setup(
    name="forte",
    version="0.1.0post2",
    url="https://github.com/asyml/forte",

    description="Forte is extensible framework for building composable and "
                "modularized NLP workflows.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache License Version 2.0',
    packages=setuptools.find_packages(
        exclude=["scripts*", "examples*", "tests*"]),
    include_package_data=True,
    platforms='any',

    install_requires=[
        'sortedcontainers==2.1.0',
        'numpy==1.16.5',
        'jsonpickle==1.4',
        'pyyaml==5.4',
        'smart-open==1.8.4',
        'typed_astunparse==2.1.4',
        'funcsigs==1.0.2',
        'mypy_extensions==0.4.3',
        'typed_ast==1.4.0',
        'jsonschema==3.0.2',
        'texar-pytorch',
        'typing>=3.7.4;python_version<"3.5"',
        'typing-inspect>=0.6.0',
        'dataclasses~=0.7;python_version<"3.7"'
    ],
    extras_require={
        'ner': ['torch>=1.1.0', 'torchtext==0.4.0', 'tqdm>=4.36.1'],
        'test': ['ddt', 'testfixtures'],
        'example': ['termcolor==1.1.0', 'hypothesis==2.0.0'],
        'wikipedia': ['rdflib==4.2.2'],
        'augment': ['transformers>=3.1'],
        'stave': ['stave']
    },
    entry_points={
        'console_scripts': [
            'generate_ontology = forte.command_line.generate_ontology.__main__:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
