# Audio Processing

## Audio DataPack
`DataPack` includes a payload for audio data and a metadata for sample rate. You can set them by calling the `set_audio` method:
```python
from forte.data.data_pack import DataPack

pack: DataPack = DataPack()
pack.set_audio(audio, sample_rate)
```
The input parameter `audio` should be a numpy array of raw waveform and `sample_rate` should be an integer the specifies the sample rate. Now you can access these data using `DataPack.audio` and `DataPack.sample_rate`.

## Audio Reader
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
