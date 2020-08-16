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
        'sortedcontainers==2.1.0',
        'numpy==1.16.5',
        'jsonpickle==1.4',
        'pyyaml==5.1.2',
        'smart-open==1.8.4',
        'typed_astunparse==2.1.4',
        'funcsigs==1.0.2',
        'mypy_extensions==0.4.3',
        'typed_ast==1.4.0',
        'jsonschema==3.0.2',
        'texar-pytorch',
        'typing>=3.7.4',
        'typing-inspect>=0.6.0',
        'dataclasses~=0.7;python_version<"3.7"'
    ],
    extras_require={
        'nltk': ['nltk==3.4.5'],
        'ner': ['torch>=1.1.0', 'torchtext==0.4.0', 'tqdm==4.36.1'],
        'srl': ['allennlp==1.0.0'],
        'sentiment': ['vaderSentiment==3.2.1'],
        'txtgen': ['regex', 'tensorflow'],
        'stanza': ['stanza==1.0.1'],
        'test': ['ddt', 'testfixtures'],
        'example': ['termcolor==1.1.0'],
        'wikipedia': ['rdflib==4.2.2'],
        'ir': ['faiss-cpu>=1.6.1', 'elasticsearch==7.5.1'],
        'spacy': ['spacy==2.3.0'],
        'allennlp': ['allennlp==1.0.0', 'allennlp-models==1.0.0',
                     'torch>=1.5.0,<1.6.0'],
        'cliner': ['marisa-trie==0.7.4', 'scipy==1.5.0',
                   'scikit-learn==0.23.1', 'repoze.lru==0.7',
                   'tensorflow-gpu==1.12.0']
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
