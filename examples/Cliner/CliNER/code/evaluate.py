######################################################################
#  CliNER - evaluate.py                                              #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Evaluate predictions of concept labels against gold.     #
######################################################################

import argparse
import os
import random
import shutil
import subprocess
import sys

from examples.Cliner.CliNER.code import tools


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='cliner evaluate')
    parser.add_argument("--predictions",
        dest="pred",
        help="Directory where predictions  are stored.",
    )
    parser.add_argument("--gold",
        dest="gold",
        help="Directory where gold standard is stored.",
    )
    parser.add_argument("--format",
        dest="format",
        help="Data format ( con ) "
    )

    args = parser.parse_args()

    if not args.pred:
        sys.stderr.write('\n\tERROR: must provide --pred argument\n\n')
        parser.print_help(sys.stderr)
        sys.stderr.write('\n')
        sys.exit()

    if not args.gold:
        sys.stderr.write('\n\tERROR: must provide --gold argument\n\n')
        parser.print_help(sys.stderr)
        sys.stderr.write('\n')
        sys.exit()

    if args.format:
        format = args.format
    else:
        sys.stderr.write('\n\tERROR: must provide --format argument\n\n')
        parser.print_help(sys.stderr)
        sys.stderr.write('\n')
        sys.exit()

    # Must specify output format
    if format not in ['i2b2']:
        sys.stderr.write('\n\tError: Must specify output format\n')
        sys.stderr.write('\tAvailable formats: i2b2\n')
        sys.stderr.write('\n')
        parser.print_help(sys.stderr)
        sys.stderr.write('\n')
        sys.exit()

    ref_files = os.listdir(args.gold)
    ref_files = map(lambda f: os.path.join(args.gold, f), ref_files)

    pred_files = os.listdir(args.pred)
    pred_files = map(lambda f: os.path.join(args.pred, f), pred_files)

    ref_files_map = tools.map_files(ref_files)
    pred_files_map = tools.map_files(pred_files)

    files = []
    for k in ref_files_map:
        if k in pred_files_map:
            files.append((pred_files_map[k], ref_files_map[k]))

    gold_list, pred_list = zip(*files)

    # print gold_list
    # print pred_list

    # create temporary directory for these files
    cliner_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')
    tempdir_name = os.path.join(tmp_dir, 'cliner_eval_%d' %
                                         random.randint(
                                             0, 256))
    # print tempdir_name

    # text_dir = os.path.join(tempdir_name, 'text/')
    pred_dir = os.path.join(tempdir_name, 'pred/')
    gold_dir = os.path.join(tempdir_name, 'gold/')

    os.mkdir(tempdir_name)
    os.mkdir(pred_dir)
    os.mkdir(gold_dir)

    # copy files
    for pred_file in pred_list:
        shutil.copy(pred_file, pred_dir)
    for gold_file in gold_list:
        shutil.copy(gold_file, gold_dir)

    # eval jar
    cliner_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(cliner_dir, 'tools', )
    eval_jar = os.path.join(eval_dir, 'i2b2va-eval.jar')

    cmd = 'java -jar %s -rcp %s -scp %s -ft con -ex all' % (
        eval_jar, gold_dir, pred_dir)
    subprocess.call(cmd, shell=True, stdout=sys.stdout)

    # cleanup after yourself
    shutil.rmtree(tempdir_name)


if __name__ == '__main__':
    main()
