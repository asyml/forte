# Audio Processing #

## Basics
### Audio DataPack
`DataPack` includes a payload for audio data and a metadata for sample rate. You can set them by calling the `set_audio` method:
```python
from forte.data.data_pack import DataPack

pack: DataPack = DataPack()
pack.set_audio(audio, sample_rate)
```
The input parameter `audio` should be a numpy array of raw waveform and `sample_rate` should be an integer the specifies the sample rate. Now you can access these data using `DataPack.audio` and `DataPack.sample_rate`.

### Audio Reader
`AudioReader` supports reading in the audio data from files under a specific directory. You can set it as the reader of your forte pipeline whenever you need to process audio files:
```python
from forte.pipeline import Pipeline
from forte.data.readers.audio_reader import AudioReader

Pipeline().set_reader(
    reader=AudioReader(),
    config={"file_ext": ".wav"}
).run(
    "path-to-audio-directory"
)
```
The example above builds a simple pipeline that can walk through the specified directory and load all the files with extension of `.wav`. `AudioReader` will create a `DataPack` for each file with the corresponding audio payload and the sample rate.


## Automatic Speech Recognition
Automatic Speech Recognition (ASR) develops methodologies and technologies that enable the recognition and translation of spoken language into text by computers. Here we using a simple example to show how to build a forte pipeline to perform speech transcription tasks. This example is based on a pretrained `wav2vec2` model and you can check out the details [here](https://huggingface.co/facebook/wav2vec2-base-960h).
## Speaker Segmentation
Speaker segmentation consists in partitioning a conversation between one or more speakers into speaker turns. It is the process of partitioning an input audio stream into acoustically homogeneous segments according to the speaker identity. A typical speaker segmentation system finds potential speaker change points using the audio characteristics. In this example, the speaker segmentation is backed by a pretrained Hugging Face model where you can find details in this [link](https://huggingface.co/pyannote/speaker-segmentation).

## Example script
For full example script, please refer to [this audio example](https://github.com/asyml/forte/tree/master/examples/audio)
