import setuptools

long_description = '''
NLP pipeline project to facilitate the production usage of NLP techniques.
'''

setuptools.setup(
    name="forte",
    version="0.0.1",
    url="https://github.com/hunterhector/forte",

    description="NLP pipeline",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(include=['forte.*', 'ft.*']),
    platforms='any',

    install_requires=[
        'sortedcontainers',
        'numpy',
        'nltk',
        'jsonpickle',
        'pyyaml',
        'deprecation',
        'smart-open',
    ],
    extras_require={
        'ner': ['pyyaml', 'torch>=1.1.0', 'torchtext', 'tqdm'],
        'srl': ['mypy-extensions', 'allennlp'],
        'txtgen': ['regex', 'tensorflow'],
        'stanfordnlp': ['stanfordnlp'],
        'test': ['ddt'],
        'example': ['termcolor'],
        'wikipedia': ['rdflib'],
        'indexer': ['faiss>=1.5.3']
    },
    package_data={
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
