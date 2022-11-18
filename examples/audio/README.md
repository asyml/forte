# Audio Processing Examples
This folder contains a series of tutorial examples that walk through the basics of building audio processing pipelines using forte.

##  Introduction

We provide a simple speech processing example here to showcase forte's capability to support a wide range of audio processing tasks. This example consists of two parts: speaker segmentation and automatic speech recognition.

### Speaker Segmentation
Speaker segmentation consists in partitioning a conversation between one or more speakers into speaker turns. It is the process of partitioning an input audio stream into acoustically homogeneous segments according to the speaker identity. A typical speaker segmentation system finds potential speaker change points using the audio characteristics. In this example, the speaker segmentation is backed by a pretrained Hugging Face model where you can find details [here](https://huggingface.co/pyannote/speaker-segmentation).

### Automatic Speech Recognition
Automatic Speech Recognition (ASR) develops methodologies and technologies that enable the recognition and translation of spoken language into text by computers. Here we using a simple example to show how to build a forte pipeline to perform speech transcription tasks. This example is based on a pretrained wav2vec2 model and you can check out the details [here](https://huggingface.co/facebook/wav2vec2-base-960h).

## Run the Example Script

This example requires **python3.8 or later versions**. Before running the script, we will need to install a few packages first:
```bash
pip install -r requirements.txt
```
Note that some packages (e.g., `soundfile`) depend on a system library called `libsndfile` which might entail [additional steps](https://pysoundfile.readthedocs.io/en/latest/#installation) for Linux users.

Now you are able to run the example script `speaker_segmentation_pipeline.py`:
```bash
python speaker_segmentation_pipeline.py
```
which will print out the annotated transcription results including speakers and their corresponding utterances. Each audio segment will be played through your PC speaker. Example output:
```
INFO:speaker_segmentation_pipeline.py:SPEAKER_01: HE JOINS US LIFE FROM THE ALLERT CENTER WITH WHAT VOTERS THINK OF TO NIGHT'S DEBATE MICHAEL
```

We include a `test_audio.wav` extracted from [VoxConverse speaker diarisation dataset](https://github.com/joonson/voxconverse) in this example. It is a conversation consisting of three speakers speaking in turns. The example script will partition the audio, transcript the waveform, and play the audio segment for each speaker. The results are not meant to be 100% accurate but they are still recognizable and reasonable.

## Code Walkthrough
The backbone of the example script is a simple forte pipeline for speech processing:
```python
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(AudioReader(), config={"file_ext": ".wav"})
    pipeline.add(SpeakerSegmentationProcessor())
    pipeline.add(AudioUtteranceASRProcessor())
    pipeline.initialize()
```
The pipeline includes three major components:
- [`AudioReader`](https://github.com/asyml/forte/blob/master/forte/data/readers/audio_reader.py) supports reading in the audio data from files under a specific directory. Use `file_ext` to configure the target file extension that you want to include as input to your pipeline. You can set it as the reader of your forte pipeline whenever you need to process audio files
- `SpeakerSegmentationProcessor` performs the speaker segmentation task utilizing a pretrained [model](https://huggingface.co/pyannote/speaker-segmentation) from HuggingFace. After partitioning the recording into segments, it creates annotations called [`AudioUtterance`](https://github.com/asyml/forte/blob/master/ft/onto/base_ontology.py#L537) to store the audio span and speaker information for later retrieval.
- `AudioUtteranceASRProcessor` transcribes audio segments into text for each `AudioUtterance` found in input datapack. It appends the transcripted text into the text payload of datapack and creates corresponding [`Utterance`](https://github.com/asyml/forte/blob/master/ft/onto/base_ontology.py#L211) with speaker identity for each segment. To illustrate the one-to-one correspondence of `AudioUtterance` and `Utterance` within each segment, it adds a [`Link`](https://github.com/asyml/forte/blob/master/forte/data/ontology/top.py#L194) entry for each speech-to-text relationship.

After running the pipeline, you can retrieve the audio and text annotations from each segment by getting all the `Link`s inside the output datapack:
```python
for asr_link in pack.get(Link):
    audio_utter = asr_link.get_parent()
    text_utter = asr_link.get_child()
```
