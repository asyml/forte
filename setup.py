import sys
import setuptools

long_description = '''
NLP pipeline project to facilitate the production usage of NLP techniques.
'''

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Forte.')

setuptools.setup(
    name="forte",
    version="0.0.1",
    url="https://github.com/asyml/forte",

    description="NLP pipeline",
    long_description=long_description,
    license='Apache License Version 2.0',
    packages=setuptools.find_packages(exclude=["scripts", "examples"]),
    include_package_data=True,
    platforms='any',

    install_requires=[
        'sortedcontainers',
        'numpy',
        'jsonpickle',
        'pyyaml',
        'deprecation',
        'smart-open',
        'typed_astunparse',
        'funcsigs',
        'mypy_extensions',
        'typed_ast',
        'jsonschema',
        'texar-pytorch',
        'typing>=3.7.4'
        'typing-inspect>=0.6.0',
    ],
    extras_require={
        'nltk': ['nltk'],
        'ner': ['pyyaml', 'torch>=1.1.0', 'torchtext>=0.4.0', 'tqdm'],
        'srl': ['mypy-extensions', 'allennlp'],
        'sentiment': ['vaderSentiment'],
        'txtgen': ['regex', 'tensorflow'],
        'stanza': ['stanza'],
        'test': ['ddt', 'testfixtures'],
        'example': ['termcolor'],
        'wikipedia': ['rdflib'],
        'ir': ['faiss-cpu>=1.6.1', 'elasticsearch'],
        'spacy': ['spacy>=2.2.3'],
        'allennlp': ['allennlp']
    },
    entry_points={
        'console_scripts': [
            'generate_ontology = scripts.generate_ontology.__main__:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ]
)
