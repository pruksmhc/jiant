# -*- coding: utf-8 -*-


"""
Loads gap coreference data and preproceses it, by 
aligning  spans from output of scripts/gap_related_scripts.py

"""
import sys
import argparse
import pandas as pd
import pickle
from enum import Enum


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

def build_spans_aligned_with_tokenization(text1, pronoun, context, text2, names):
    """
    Builds the alignment while also tokenizing the input piece by piece. 
    Gets the indices of both name1 and name2 in ALL PLACES. 

    Output:
    tokenized text, [ (tokenized_start, tokenized_end, type_of_name (1, 2, or pronoun))] for the number of times 
    it occurs in the text 

    """
    import math
    import re
    total_text = text1 + " "+ pronoun +" "+ text2
    total_text= total_text.replace("\\n"," ")
    lower_text = total_text.lower()
    lower_text = re.sub(' +', ' ', lower_text)
    total_text =  re.sub(' +', ' ', total_text)
    total_name_indices = []
    for name in names:
        name1 = name.replace("\\n", " ")
        name1 = re.sub(' +', ' ',name1).strip().lower()
        for f_name in re.finditer(r'\b'+name1+r'\b', lower_text):
        	total_name_indices.append((f_name.start(), f_name.end(), "name" + name))
    if isinstance(context, str) is False:
        pronoun = pronoun.replace("\\n", " ") 
        context = re.sub(' +', ' ',pronoun).strip().lower()
    else:
        context = re.sub(' +', ' ', context.replace("\\n","")).strip().lower()
    prn_name = re.search(context, str(lower_text))
    total_name_indices.append((prn_name.start(), prn_name.start() + len(pronoun), "pronoun"))
    sorted_indices = sorted(total_name_indices, key=lambda x: x[0])
    new_spans = []
    text = total_text
    current_tokenization = []
    text1 = text[:sorted_indices[0][0]]
    new_tokens = text1.split()
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)
    span1 = text[sorted_indices[0][0]:sorted_indices[0][1]]
    span_tokens = span1.split()
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization)
    new_spans.append((new_span1start, new_span1end, sorted_indices[0][2])) 
    for i in range(1, len(sorted_indices)):
    	current_text = text[sorted_indices[i-1][1] : sorted_indices[i][0]]
    	# get text in between
    	new_tokens = current_text.split()
    	current_tokenization.extend(new_tokens)
    	new_spanstart = len(current_tokenization)
    	span = text[sorted_indices[i][0]:sorted_indices[i][1]]
    	span_tokens = span.split()
    	current_tokenization.extend(span_tokens)
    	new_spanend = len(current_tokenization)
    	new_spans.append((new_spanstart, new_spanend, sorted_indices[i][2])) 
    final_text = text[sorted_indices[-1][1]:]
    new_tokens = final_text.split()
    current_tokenization.extend(new_tokens)
    new_tokens = " ".join(text.split())
    return  new_tokens, new_spans

def align_spans(file_name,  data_dir):
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
    gap_text = pd.read_csv(data_dir+ file_name + ".csv",  header = 0)
    new_pandas = []
    for i in range(len(gap_text)):
        row = gap_text.iloc[i]
        text = row['text__txt1']
        pronoun = row["text__pron"]
        text_2 = row["text__txt2"]
        context = row["quote__quote2"]
        name1 = row["answers__answer__001"]
        name2 = row["answers__answer__002"]
        label = row["correctAnswer"]
        try:
            text, new_spans = build_spans_aligned_with_tokenization(text, pronoun, context, text_2,  [name1, name2] )
        except:
            import pdb; pdb.set_trace()
            continue
        # then, now we organize to becom
        # for each 
        # just try an LSTM model. 
        name_a_indices =  [span[:2] for span in new_spans if span[2] =="name"+ name1]
        name_b_indices = [span[:2] for span in new_spans if span[2] =="name"+name2]
        prn_index = [span[:2] for span in new_spans if span[2] =="pronoun"]
        if label.strip() == "A":
        	label = True
        else:
        	label = False
        new_pandas.append([text, pronoun, prn_index, name1, name_a_indices, label])
        new_pandas.append([text, pronoun, prn_index, name2, name_b_indices, not label])
    result = pd.DataFrame(new_pandas, columns=["text", "pronoun", "pronoun_indices", "NP_in_question", "NP_in_question_indices", "label"])
    import pdb; pdb.set_trace()
    length_results = result["NP_in_question_indices"].apply(lambda x:len(x))
    import numpy as np 
    mask = np.where(length_results > 0)
    result = result.loc[mask]
    result.to_csv(data_dir+file_name+"final", sep="\t")

def parse_weird(split, file_name, data_dir):
    # this is for resource-source3 in data/WNLI
    import pdb; pdb.set_trace()
    gap_text = pd.read_csv(data_dir+ file_name + ".csv",  header = None)
    new_pandas = []
    for i in range(0, len(gap_text), 3):
        try: 
            current_lines = gap_text.iloc[i:i+3]
            text = current_lines.iloc[0][0]
            pronoun = current_lines.iloc[1][0].split(",")[0].replace('”', '').replace('“', '')
            context = current_lines.iloc[1][0].split(",")[1].split(":")[0].replace('”', '').replace('“', '')
            options = current_lines.iloc[1][0].split("”,")[1].split(")")
            names = [name.split("(")[0].strip() for name in options[1:]]
            text1 = text.split(context)[0] 
            if len(text.split(context)) == 1:
                text2 = ""
            else:
                text2 = text.split(context)[1]
            # and now separate by ) 
            label = current_lines.iloc[2][0].split(")")[1].strip()
            text, new_spans = build_spans_aligned_with_tokenization(text1, pronoun, context, text2, names)
            prn_index = [span[:2] for span in new_spans if span[2] == "pronoun"]
            for name in names:
                indices = [span[:2] for span in new_spans if span[2] == "name"+name ]
                if label == name:
                    new_pandas.append([text, pronoun, prn_index, name, indices, True])
                else:
                    new_pandas.append([text, pronoun, prn_index, name, indices, False])
        except:
            import pdb; pdb.set_trace()
    result = pd.DataFrame(new_pandas, columns=["text", "pronoun", "pronoun_indices", "NP_in_question", "NP_in_question_indices", "label"])
    length_results = result["NP_in_question_indices"].apply(lambda x:len(x))
    import numpy as np 
    mask = np.where(length_results > 0)
    result = result.loc[mask]
    import json
    import pdb; pdb.set_trace()
    result.to_csv(data_dir+file_name+"final", sep="\t")

def parse_test():
    tokenizer_name = "bert-base-cased"
    gap_text = pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/test.tsv", sep="\t",  header = 0)
    new_pandas = []
    for i in range(len(gap_text)):
        row = gap_text.iloc[i]
        text = row['text1']
        pronoun = row["pronoun"]
        text_2 = row["text2"]
        context = row["quote"]
        name1 = row["answer"]
        label = row["gold"]
        try:
            text, new_spans = build_spans_aligned_with_tokenization(text, pronoun, context, text_2,  [name1] )
        except:
            import pdb; pdb.set_trace()
            continue
        # then, now we organize to becom
        # for each 
        # just try an LSTM model. 
        name_a_indices =  [span[:2] for span in new_spans if span[2] =="name"+ name1]
        prn_index = [span[:2] for span in new_spans if span[2] =="pronoun"]
        if label == "No":
            label = False
        else:
            label = True
        new_pandas.append([text, pronoun, prn_index, name1, name_a_indices, label])
    result = pd.DataFrame(new_pandas, columns=["text", "pronoun", "pronoun_indices", "NP_in_question", "NP_in_question_indices", "label"])
    import pdb; pdb.set_trace()
    length_results = result["NP_in_question_indices"].apply(lambda x:len(x))
    import numpy as np 
    mask = np.where(length_results > 0)
    result = result.loc[mask]
    result.to_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/testfinal" ,sep="\t")

def convert_to_json():
    import json
    ref = "/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/"
    for link in [ "test_final.tsv"]:
        tokenizer_name = "bert-base-cased"
        result = pd.read_csv(ref+link, sep="\t")
        json_writer = open(ref + link +".json", "w", encoding="utf8")
        total = []
        from ast import literal_eval 
        import pdb; pdb.set_trace()
        for idx, result in result.iterrows():
            val = {"text": result["text"],  "targets": [{"label": [str(result["label"])], "span1_text": result["NP_in_question"], "span1": literal_eval(result["NP_in_question_indices"])[0], "span2_text": result["pronoun"], "span2": literal_eval(result["pronoun_indices"])[0]}]}
            total.append(val)
        json.dump(total, json_writer)
        json_writer.close()


def move_yes_val_to_train():
    val =  pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/val.tsv", sep="\t")
    train = pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/train.tsv", sep="\t")
    test =  pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/test_final.tsv", sep="\t")
    def get_distribution(data):
        yes_label = len(data["label"].nonzero()[0])
        return yes_label/len(data)
    # take the distribuiton from yes to no 
    import numpy as np
    indices = val["label"].nonzero()[0]
    np.random.shuffle(indices)
    i = 0 
    potential_yes_val = val[val["label"] == True]
    set_of_hyps = list(set(potential_yes_val.text))
    np.random.shuffle(set_of_hyps)
    import pdb; pdb.set_trace()
    while abs(get_distribution(val) - get_distribution(test)) > 0.01 and i < len(set_of_hyps):
        print(i)
        selected_hyp = set_of_hyps[i]
        i += 1
        to_move = val[val["text"] == selected_hyp]
        is_okay = to_move[to_move["label"] == False]
        val = val[val["text"] != selected_hyp]
        val = val.append(is_okay)
        print(len(val))
    print(get_distribution(val))
    print(get_distribution(test))
    import pdb; pdb.set_trace()
    val.to_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/val_same_distribution_test.tsv", sep="\t")
    train.to_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/train_after_redistribution.tsv", sep="\t")
    # want it to be equal to test set, which has 65% distribuiton. 
    import pdb; pdb.set_trace()


def get_train_test_split():
    import pdb; pdb.set_trace()
    tt1 = pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/WNLI/resultfinal", sep="\t")
    tt2 = pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/WNLI/result-source2final", sep="\t")
    tt3 = pd.read_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/WNLI/result-source3final", sep="\t") 
    tt1 = tt1.append(tt2)
    tt1 = tt1.append(tt3)
    # this one is in the test set:
    bad_indices = tt1[tt1["text"].str.contains("A number of times")].index
    tt1 = tt1.drop(bad_indices)
    set_of_hyps = list(set(tt1.text))
    import numpy as np
    msk = np.random.rand(len(set_of_hyps)) < 0.8
    train_indices = msk.nonzero()[0]
    train = pd.DataFrame()
    val = pd.DataFrame()
    import pdb; pdb.set_trace()
    for index in train_indices:
        hyp = set_of_hyps[index]
        train =  train.append(tt1[tt1["text"] == hyp])
    val_msk = ~msk
    val_indices = val_msk.nonzero()[0]
    for index in val_indices:
        hyp = set_of_hyps[index]
        val = val.append(tt1[tt1["text"] == hyp])
    import pdb; pdb.set_trace()
    train.to_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/train.tsv", sep="\t")
    val.to_csv("/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/winograd-coref/val.tsv", sep="\t")

    # random 90-10 split. 

    # wnli test. 

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
    parser.add_argument(
        '-f',
        '--file_name',
        help='file name',
        type=str,
        default='MosesTokenizer')
    args = parser.parse_args(arguments)
    #convert_to_json()
    #get_train_test_split()
    # parse_test()
    #convert_to_json()
    data_dir = "/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data/WNLI/"
    file_name = "result"
    #get_train_test_split()
    move_yes_val_to_train()
    #align_spans(file_name, data_dir)
    #parse_weird("test",file_name,data_dir)
    #get_train_test_split()
    #convert_to_json()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
