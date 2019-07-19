import setuptools

long_description = '''
NLP pipeline project to facilitate the production usage of NLP techniques.
'''

setuptools.setup(
    name="nlp",
    version="0.0.1",
    url="https://gitlab.int.petuum.com/internal/aml/nlp_pipeline",

    description="NLP pipeline",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'sortedcontainers',
        'numpy',
        'nltk',
        'jsonpickle'
    ],
    extras_require={
        'ner': ['pyyaml', 'torch', 'torchtext', 'tqdm'],
        'srl': ['mypy-extensions'],
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
    ],
)
