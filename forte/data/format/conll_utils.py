"""
Here we define some utility functions for CoNLL evaluation format change.
We can add other format conversion function for CoNLL here in the future.
"""


def write_tokens_to_file(pred_pack, pred_request, refer_pack, refer_request,
                         output_filename):
    opened_file = open(output_filename, "w+")
    for pred_sentence, tgt_sentence in zip(
            pred_pack.get_data(**pred_request),
            refer_pack.get_data(**refer_request)
    ):

        pred_tokens, tgt_tokens = (
            pred_sentence["Token"],
            tgt_sentence["Token"],
        )
        for i in range(len(pred_tokens["text"])):
            w = tgt_tokens["text"][i]
            p = tgt_tokens["pos_tag"][i]
            ch = tgt_tokens["chunk_tag"][i]
            tgt = tgt_tokens["ner_tag"][i]
            pred = pred_tokens["ner_tag"][i]

            opened_file.write(
                "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred)
            )

        opened_file.write("\n")
    opened_file.close()
