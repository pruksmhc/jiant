#!/usr/bin/env python

# Script to convert Noun-Verb ambiguity data
# (Elkahky et al. 2018, https://aclweb.org/anthology/D18-1277)
# to edge probing format.
#
# Download the data from
# https://github.com/google-research-datasets/noun-verb
#
# Usage:
#   ./convert-noun-verb.py \
#       -i /path/to/noun-verb/<filename>.conll \
#       -o /path/to/probing/data/noun-verb/<filename>.json
#
# Note that the paper uses the fine-grained tags (in train.conll) for training,
# then evaluates by mapping the tags onto VERB/NON-VERB categories. The output
# of this script uses only VERB/NON-VERB as the target labels, but includes
# the fine-grained tags in targets.info when available.

import sys
import os
import json
import collections
import argparse

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import utils
import pandas as pd
from tqdm import tqdm

import conllu

from typing import Dict, Tuple, List

def conll_to_record(sentence: List[Dict]) -> Dict:
    tokens = [item['form'] for item in sentence]

    record = {}
    record['text'] = " ".join(tokens)
    record['targets'] = []
    for i, item in enumerate(sentence):
        feats = item['feats']
        if feats is None:
            continue
        target = {}
        target['label'] = feats['POS']
        target['span1'] = [i, i+1]
        target['info'] = {'upostag': item['upostag'],
                          'xpostag': item['xpostag']}
        record['targets'].append(target)

    return record

def convert_file(fname: str, target_fname: str):
    with open(fname) as fd:
        sentences = conllu.parse(fd.read())

    records = map(conll_to_record, sentences)
    records = tqdm(records, total=len(sentences))
    utils.write_file_and_print_stats(records, target_fname)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', type=str, required=True,
                        help="Input .tsv file.")
    parser.add_argument('-o', dest='output', type=str, required=True,
                        help="Output .json file.")
    args = parser.parse_args(args)

    pd.options.display.float_format = '{:.2f}'.format
    log.info("Converting %s", args.input)
    convert_file(args.input, args.output)


if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)



