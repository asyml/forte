# Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compose paraphrased sentences back to paragraphs.
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_file', type=str, default='backward_gen.txt',
    help="Path to the back translated sentence file.")
parser.add_argument(
    '--output_file', type=str, default='back_translate.txt',
    help="Path to the output paragraph file.")
parser.add_argument(
    '--doc_len_file', type=str, default='train_doc_len.json',
    help="The file that records the length information.")
args = parser.parse_args()


def main():
    with open(args.input_file, encoding='utf-8') as inf:
        sentences = inf.readlines()
    with open(args.doc_len_file, encoding='utf-8') as inf:
        doc_len_list = json.load(inf)
    cnt = 0
    print("Printing paraphrases:")
    with open(args.output_file, "w", encoding='utf-8') as ouf:
        for i, sent_num in enumerate(doc_len_list):
            para = ""
            for _ in range(sent_num):
                para += sentences[cnt].strip() + " "
                cnt += 1
            print("Paraphrase {}: {}".format(i, para))
            ouf.write(para.strip() + "\n")


if __name__ == '__main__':
    main()
