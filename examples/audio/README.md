# Audio Processing Examples
This folder contains a series of tutorial examples that walk through the basics of building audio processing pipelines using forte.

## Automatic Speech Recognition
Automatic Speech Recognition (ASR) develops methodologies and technologies that enable the recognition and translation of spoken language into text by computers. Here we using a simple example to show how to build a forte pipeline to perform speech transcription tasks. This example is based on a pretrained wav2vec2 model and you can check out the details [here](https://huggingface.co/facebook/wav2vec2-base-960h).

### Run the Demo Code

An audio file processing package called `soundfile` is required for this example. You can run 
```bash
pip install soundfile
```
or
```bash
pip install forte[audio_ext]
```
Note that [additional steps](https://pysoundfile.readthedocs.io/en/latest/#installation) might apply to Linux users.

You will also need to install `transformers` for the pretrained model from HuggingFace.
```bash
pip install transformers
```
Now you are able to run the example script `asr_pipeline.py`:
```bash
python asr_pipeline.py
```
which will print out the annotated transcription results including speaker, text, the size of audio payload, and the sample rate. Example output:
```
INFO:asr_pipeline.py:speaker: NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
INFO:asr_pipeline.py:Size of audio payload: (77040,)
INFO:asr_pipeline.py:Sample rate: 16000
```
