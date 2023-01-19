This is a very simple serialization demo that use the built-in JSON serializer.

First, let's install some simple processors via:

`
pip install forte.nltk
`

To ensure you are using the current version of Forte, go to Forte root and install from source:

`
cd <forte source directory>
pip install .
`

Then just run the following command from this example directory:

`
python serialize_example.py "../../data_samples/ontonotes/00/"
`

You should be able to see the progress and the serialized content.