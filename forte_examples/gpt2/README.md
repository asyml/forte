# Text Generation Pipeline


In this example, we show the use of

1) Use of a MultiPack pipeline
2) Loading a pipeline from config file
3) Using GPT2 processor to generate text and to run analysis on them.

`sample_multipack_pipeline_gpt.yml` specifies the reader and the processors in this pipeline. Here
we use `MultiPackSentenceReader` to read data from text file into a multipack. This example
`data/sample.text` loads it into a `MultiPack` where each line is used as the initial context for
GPT2. In the multipack, the input sentences and the generated sentences are keyed by `input_src`
and `output_tgt` respectively. We run `NLTKWordTokenizer` and `NLTKPOSTagger` on the generated
sentences. To see this in action, run the following command

```bash
python multipack_pipeline_gpt2.py
```

You will be able to see the original sentence, generated sentence along with POS tags for each word
of the generated sentence. Additionally, the multipack will be saved to `multipack_output.txt`.
