
"""
Loads gap coreference data and preproceses it, by 
aligning  spans from output of scripts/gap_related_scripts.py

"""

import sys
import argparse
import pandas as pd
import pickle
from enum import Enum
import logging as log
import retokenize_edge_data as retokenize 

UNKNOWN = "Unknown"
MASCULINE = "Male"
FEMININE = "Female"


# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': FEMININE,
    'her': FEMININE,
    'hers': FEMININE,
    'he': MASCULINE,
    'his': MASCULINE,
    'him': MASCULINE,
}

def getEnd(index, word):
    return index  + len(word)

def build_spans_aligned_with_tokenization(text, tokenizer_name, orig_span1_start, orig_span1_end, orig_span2_start, orig_span2_end):
    """
    Builds the alignment while also tokenizing the input piece by piece. 
    """
    if orig_span1_end > orig_span2_start:
        # switch them since the pronoun comes after 
        span2_start = orig_span1_start
        span2_end = orig_span1_end
        span1_start = orig_span2_start
        span1_end = orig_span2_end
    else:
        span1_start = orig_span1_start
        span1_end = orig_span1_end
        span2_start = orig_span2_start
        span2_end = orig_span2_end

    current_tokenization = []
    text = text.split()
    aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    text1 = " ".join(text[:span1_start])
s    ta, new_tokens = aligner_fn(text1)
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)

    span1 = " ".join(text[span1_start:span1_end])
    ta, span_tokens = aligner_fn(span1)
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization) 
    text2 = " ".join(text[span1_end:span2_start])
    ta, new_tokens = aligner_fn(text2)
    current_tokenization.extend(new_tokens)
    new_span2start = len(current_tokenization)

    span2 = " ".join(text[span2_start:span2_end])
    ta, span_tokens = aligner_fn(span2)
    current_tokenization.extend(span_tokens)
    new_span2end = len(current_tokenization)

    text3 = " ".join(text[span2_end:])
    ta, span_tokens = aligner_fn(text3)
    current_tokenization.extend(span_tokens)

    text = " ".join(current_tokenization)   
    if orig_span1_end > orig_span2_start:
        return new_span2start, new_span2end, new_span1start, new_span1end,text
    return  new_span1start, new_span1end, new_span2start, new_span2end, text

def align_spans(split, tokenizer_name, data_dir):
    """
        This processes the dataset into the form that edge probing can read in
        and aligns the spain indices from character to tokenizerspan-aligned indices. 
        For example, 
         "I like Bob Sutter yeah " becomes soemthing like
        ["I", "like", "Bob", "Sut", "ter", "yeah"]
        The gold labels given in GAP ha
        s noun index as [7:16], however, with tokenization, we
        want tokenization of [2:4]

        Output: 
        A TSV file readable
    """
    gap_text = pd.read_csv(data_dir+"gap-" + split + ".tsv",  header = 0, delimiter="\t")
    new_pandas = []
    for i in range(len(gap_text)):
        row = gap_text.iloc[i]
        text = row['Text']
        pronoun = row['Pronoun']
        gender = PRONOUNS[pronoun.lower()]
        orig_pronoun_index = row["Pronoun-offset"]
        orig_end_index_prnn = getEnd(orig_pronoun_index, pronoun)
        orig_first_index = row["A-offset"]
        first_word = row["A"]
        orig_end_index = getEnd(orig_first_index, first_word)
        pronoun_index, end_index_prnn, first_index, end_index,text = build_spans_aligned_with_tokenization(text, tokenizer_name, orig_pronoun_index, orig_end_index_prnn, orig_first_index, orig_end_index)
        
        label = str(row["A-coref"]).lower()
        new_pandas.append([text, gender, pronoun_index, end_index_prnn, first_index, end_index, label])

        second_index = row["B-offset"]
        second_word = row["A"]
        end_index_b = getEnd(second_index, second_word)
        _, _, second_index, end_index_b, text = build_spans_aligned_with_tokenization(text, tokenizer_name, orig_pronoun_index, orig_end_index_prnn, second_index, end_index_b)
        label_b = str(row['B-coref']).lower()

        new_pandas.append([text, gender, pronoun_index, end_index_prnn, second_index, end_index_b, label_b])

    result = pd.DataFrame(new_pandas, columns=["text", "gender", "prompt_start_index", "prompt_end_index", "candidate_start_index", "candidate_end_index", "label"])
    result.to_csv(data_dir+"processed/gap-coreference/__"+split+"__.retokenized."+tokenizer_name, sep="\t")

def align_winograd(tokenizer_name):
    import json
    # I want to actually amke this stuff right now. 
    link = "/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/"
    for name in ["test_final.tsv.json", "train_final.tsv.json", "val_final.tsv.json"]:
        fname = link + name
        json_writer = open(fname + "retokenized"+str(tokenizer_name), "w", encoding="utf8")
        inputs = list(pd.read_json(fname).T.to_dict().values())
        for i in inputs:
            text = i["text"]
            orig_pronoun_index = i["targets"][0]["span2"][0]
            orig_end_index_prnn =  i["targets"][0]["span2"][1]
            log.info(i)
            second_index = i["targets"][0]["span1"][0]
            end_index_b = i["targets"][0]["span1"][1]
            pronoun_index, end_index_prnn, first_index, end_index,text =  build_spans_aligned_with_tokenization(text, tokenizer_name, orig_pronoun_index, orig_end_index_prnn, second_index, end_index_b)
            log.info(text)
            log.info("PRONOUN")
            log.info(pronoun_index)
            log.info(end_index_prnn)
            log.info("actual np")
            log.info(str(first_index))
            log.info(str(end_index))
            i["targets"][0]["span2"] = [pronoun_index, end_index_prnn]
            i["text"] = text
            i["targets"][0]['span1'] = [first_index, end_index] 
            json.dump(i, json_writer)
            json_writer.write("\n")
        json_writer.close()


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_dir',
        help='directory to save data to',
        type=str,
        default='../data')
    parser.add_argument(
        '-t',
        '--tokenizer',
        help='intended tokenization',
        type=str,
        default='MosesTokenizer')
    align_winograd("bert-large-uncased")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
