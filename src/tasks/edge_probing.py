"""Task definitions for edge probing."""
import collections
import itertools
import json
import logging as log
import os

import numpy as np

from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure
from ..allennlp_mods.correlation import FastMatthews
from sklearn.metrics import f1_score
from allennlp.data.tokenizers import Token
# Fields for instance processing
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, \
    SpanField, ListField, MetadataField
from ..allennlp_mods.multilabel_field import MultiLabelField
from allennlp.data import vocabulary

from ..utils import serialize
from ..utils import utils
from ..utils import data_loaders

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import Task, sentence_to_text_field, create_subset_scorers, collect_subset_scores, update_subset_scorers
from .registry import register_task, REGISTRY  # global task registry

class MacroF1():
  def __init__(self):
    self.f1_pos_class = F1Measure(positive_label=1)
    self.f1_neg_class = F1Measure(positive_label=0)

  def __call__(self, logits, labels):
    self.f1_pos_class(logits, labels)
    self.f1_neg_class(logits, labels)

  def get_metric(self, reset=False):
    macro_f1 = (self.f1_pos_class.get_metric(reset)[2] + self.f1_neg_class.get_metric(reset)[2]) / 2.0
    return macro_f1

# you should just get the F1 scores so fa
# THe macro average is() (0.1 + 0.5 / 2) +(0.3 + 0.7/ 2))/2 = 0.1 + 0.5 + 0.3 + 0.7 / 4
# and (0.2+0.8)
##
# Class definitions for edge probing. See below for actual task registration.
class EdgeProbingTask(Task):
    ''' Generic class for fine-grained edge probing.

    Acts as a classifier, but with multiple targets for each input text.

    Targets are of the form (span1, span2, label), where span1 and span2 are
    half-open token intervals [i, j).

    Subclass this for each dataset, or use register_task with appropriate kw
    args.
    '''
    @property
    def _tokenizer_suffix(self):
        ''' Suffix to make sure we use the correct source files,
        based on the given tokenizer.
        '''
        if self.tokenizer_name:
            return ".retokenized." + self.tokenizer_name
        else:
            return ""

    def tokenizer_is_supported(self, tokenizer_name):
        ''' Check if the tokenizer is supported for this task. '''
        # Assume all tokenizers supported; if retokenized data not found
        # for this particular task, we'll just crash on file loading.
        return True

    def __init__(self, path: str, max_seq_len: int,
                 name: str,
                 label_file: str = None,
                 files_by_split: Dict[str, str] = None,
                 is_symmetric: bool = False,
                 single_sided: bool = False, **kw):
        """Construct an edge probing task.

        path, max_seq_len, and name are passed by the code in preprocess.py;
        remaining arguments should be provided by a subclass constructor or via
        @register_task.

        Args:
            path: data directory
            max_seq_len: maximum sequence length (currently ignored)
            name: task name
            label_file: relative path to labels file
            files_by_split: split name ('train', 'val', 'test') mapped to
                relative filenames (e.g. 'train': 'train.json')
            is_symmetric: if true, span1 and span2 are assumed to be the same
                type and share parameters. Otherwise, we learn a separate
                projection layer and attention weight for each.
            single_sided: if true, only use span1.
        """
        super().__init__(name, **kw)

        assert files_by_split is not None
        self.max_seq_len = max_seq_len
        self._files_by_split = {
            split: os.path.join(path, fname) + self._tokenizer_suffix
            for split, fname in files_by_split.items()
        }
        self._iters_by_split = self.load_data()
        self.is_symmetric = is_symmetric
        self.single_sided = single_sided

        label_file = os.path.join(path, label_file)
        self.all_labels = list(utils.load_lines(label_file))
        self.n_classes = len(self.all_labels)
        # see add_task_label_namespace in preprocess.py
        self._label_namespace = self.name + "_labels"

        # Scorers
        #  self.acc_scorer = CategoricalAccuracy()  # multiclass accuracy
        self.mcc_scorer = FastMatthews()
        self.acc_scorer = BooleanAccuracy()  # binary accuracy
        self.f1_scorer = F1Measure(positive_label=1)  # binary F1 overall
        self.val_metric = "%s_f1" % self.name  # TODO: switch to MCC?
        self.val_metric_decreases = False

    def _stream_records(self, filename):
        skip_ctr = 0
        total_ctr = 0
        for record in utils.load_json_data(filename):
            total_ctr += 1
            # Skip records with empty targets.
            # TODO(ian): don't do this if generating negatives!
            if not record.get('targets', None):
                skip_ctr += 1
                continue
            record["text"] = " ".join(record["text"].split()[:510])
            yield record
        log.info("Read=%d, Skip=%d, Total=%d from %s",
                 total_ctr - skip_ctr, skip_ctr, total_ctr,
                 filename)

    @staticmethod
    def merge_preds(record: Dict, preds: Dict) -> Dict:
        """ Merge predictions into record, in-place.

        List-valued predictions should align to targets,
        and are attached to the corresponding target entry.

        Non-list predictions are attached to the top-level record.
        """
        record['preds'] = {}
        for target in record['targets']:
            target['preds'] = {}
        for key, val in preds.items():
            if isinstance(val, list):
                assert len(val) == len(record['targets'])
                for i, target in enumerate(record['targets']):
                    target['preds'][key] = val[i]
            else:
                # non-list predictions, attach to top-level preds
                record['preds'][key] = val
        return record

    def load_data(self):
        iters_by_split = collections.OrderedDict()
        for split, filename in self._files_by_split.items():
            #  # Lazy-load using RepeatableIterator.
            #  loader = functools.partial(utils.load_json_data,
            #                             filename=filename)
            #  iter = serialize.RepeatableIterator(loader)
            iter = list(self._stream_records(filename))
            iters_by_split[split] = iter
        return iters_by_split

    def get_split_text(self, split: str):
        ''' Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return self._iters_by_split[split]

    def get_num_examples(self, split_text):
        ''' Return number of examples in the result of get_split_text.

        Subclass can override this if data is not stored in column format.
        '''
        return len(split_text)

    def _make_span_field(self, s, text_field, offset=0):
        return SpanField(s[0] + offset, s[1] - 1 + offset, text_field)

    def _pad_tokens(self, tokens):
        """Pad tokens according to the current tokenization style."""
        if self.tokenizer_name.startswith("bert-"):
            # standard padding for BERT; see
            # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py#L85
            return ["[CLS]"] + tokens + ["[SEP]"]
        else:
            return [utils.SOS_TOK] + tokens + [utils.EOS_TOK]

    def make_instance(self, record, idx, indexers) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record['text'].split()  # already space-tokenized by Moses
        tokens = self._pad_tokens(tokens)
        text_field = sentence_to_text_field(tokens, indexers)

        d = {}
        d["idx"] = MetadataField(idx)

        d['input1'] = text_field
        d['span1s'] = ListField([self._make_span_field(t['span1'], text_field, 1)
                                 for t in record['targets']])
        if not self.single_sided:
            d['span2s'] = ListField([self._make_span_field(t['span2'], text_field, 1)
                                     for t in record['targets']])

        d['span2s'] = sentence_to_text_field(record['targets'][0]["span2"].split(), indexers)
        # Always use multilabel targets, so be sure each label is a list.
        labels = [utils.wrap_singleton_string(str(t['label']).lower())
                  for t in record['targets']]
        d['labels'] = ListField([MultiLabelField(label_set, label_namespace=self._label_namespace, skip_indexing=False) for label_set in labels])
        return Instance(d)

    def process_split(self, records, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _map_fn(r, idx): return self.make_instance(r, idx, indexers)
        return map(_map_fn, records, itertools.count())

    def get_all_labels(self) -> List[str]:
        return self.all_labels

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split, iter in self._iters_by_split.items():
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for record in iter:
                yield record["text"].split()

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        metrics = {}
        metrics['mcc'] = self.mcc_scorer.get_metric(reset)
        metrics['acc'] = self.acc_scorer.get_metric(reset)
        precision, recall, f1 = self.f1_scorer.get_metric(reset)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        return metrics

    def update_subset_metrics(logits, labels, tagmask=None):
      return

@register_task('ultrafine', rel_path = 'ultrafine')
class UltrafinedCoreferenceTask(EdgeProbingTask):
    def __init__(self, path, domain=["general", "fine", "finer"], single_sided=False, **kw):
        self._domain_namespace = kw["name"] + "_tags"
        self._files_by_split = {'train': "final_parsed_train.json", 'val': 'final_parsed_val.json', "test": "final_parsed_test.json"}
        # current new one. 
        self.macro_f1_scorer = MacroF1()
        self.micro_f1_scorer = F1Measure(positive_label=1) #micro average
        self.scorers = [self.macro_f1_scorer, self.micro_f1_scorer]
        self.domains = domain
        self.tag_list = domain
        num_domains = 3
        self.micro_subset_scorers = create_subset_scorers(num_domains, F1Measure, positive_label=1)
        self.macro_subset_scorers = create_subset_scorers(num_domains, MacroF1)
        super().__init__(files_by_split=self._files_by_split, label_file="labels.txt", path=path, single_sided=True, **kw)

    def make_instance(self, record, idx, indexers) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record['text'].split()  # already space-tokenized by Moses
        tokens = self._pad_tokens(tokens)
        type_label_tokens = record["targets"][0]['span2']
        type_length = len(type_label_tokens)
    
        offset = len(tokens) + type_length - 512
        # if the length of tokens and the appended type is too much.
        if len(tokens) + type_length > 512:
          span_indices = record["targets"][0]['span1']
          # make sure you strip the part of the text that doesn't contain the 
          # span
          if span_indices[0] < len(tokens) - offset:
            # remove the right end of the text
            tokens = tokens[:len(tokens) - offset - 1] + [tokens[-1]]
          else:
            # remove left end
            tokens = tokens[0] + tokens[len(tokens) - offset + 1:]
        tokens.extend(type_label_tokens)# append to become [CLS] text [SEP] label
        text_field = sentence_to_text_field(tokens, indexers) # add to the vocabulary

        d = {}
        d["idx"] = MetadataField(idx)

        d['input1'] = text_field
        d['span1s'] = ListField([self._make_span_field(t['span1'], text_field, 1)
                                 for t in record['targets']])
        if not self.single_sided:
            d['span2s'] = ListField([self._make_span_field(t['span2'], text_field, 1)
                                     for t in record['targets']])

        d['span2s'] = sentence_to_text_field(record['targets'][0]["span2"], indexers)
        # Always use multilabel targets, so be sure each label is a list.
        labels = [utils.wrap_singleton_string(str(t['label']).lower())
                  for t in record['targets']]
        d['labels'] = ListField([MultiLabelField(label_set, label_namespace=self._label_namespace, skip_indexing=False) for label_set in labels])
        return Instance(d)

    def process_split(self, records, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _map_fn(r, idx): 
          instance = self.make_instance(r, idx, indexers)
          tag_field = MultiLabelField([r["targets"][0]["type_category"]], label_namespace=self._domain_namespace)
          instance.add_field("tagmask", field=tag_field)
          return instance
        return map(_map_fn, records, itertools.count())

    def update_subset_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        binary_scores = torch.stack([-1 * logits, logits], dim=2)
        if tagmask is not None:
             update_subset_scorers(self.micro_subset_scorers, binary_scores, labels, tagmask)
             update_subset_scorers(self.macro_subset_scorers, binary_scores, labels, tagmask)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split, iter in self._iters_by_split.items():
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for record in iter:
                yield record["text"].split()
                yield record["targets"][0]['span2']

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split, iter in self._iters_by_split.items():
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for record in iter:
                yield record["text"].split()
                yield record["targets"][0]['span2']

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        f1 = self.f1_scorer.get_metric(reset)
        micro_f1 = f1[0]
        macro_f1 = f1[1]
        collected_metrics = {"overall_micro_f1": micro_f1, "overall_macro_f1": macro_f1}
        for score_name, scorer in list(self.subset_scorers.items()):
          collected_metrics.update(collect_subset_scores(scorer, score_name, self.domains, reset))
        return collected_metrics

##
# Core probing tasks. as featured in the paper.
##
# Part-of-Speech tagging on OntoNotes.
register_task('edges-pos-ontonotes',
               rel_path='edges/ontonotes-constituents',
               label_file="labels.pos.txt", files_by_split={
                   'train': "consts_ontonotes_en_train.pos.json",
                   'val': "consts_ontonotes_en_dev.pos.json",
                   'test': "consts_ontonotes_en_test.pos.json",
               }, single_sided=True)(EdgeProbingTask)
# Constituency labeling (nonterminals) on OntoNotes.
register_task('edges-nonterminal-ontonotes',
               rel_path='edges/ontonotes-constituents',
               label_file="labels.nonterminal.txt", files_by_split={
                   'train': "consts_ontonotes_en_train.nonterminal.json",
                   'val': "consts_ontonotes_en_dev.nonterminal.json",
                   'test': "consts_ontonotes_en_test.nonterminal.json",
               }, single_sided=True)(EdgeProbingTask)
# Dependency edge labeling on English Web Treebank (UD).
register_task('edges-dep-labeling-ewt', rel_path='edges/dep_ewt',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Entity type labeling on OntoNotes.
register_task('edges-ner-ontonotes',
               rel_path='edges/ontonotes-ner',
               label_file="labels.txt", files_by_split={
                   'train': "ner_ontonotes_en_train.json",
                   'val': "ner_ontonotes_en_dev.json",
                   'test': "ner_ontonotes_en_test.json",
               }, single_sided=True)(EdgeProbingTask)
# SRL CoNLL 2012 (OntoNotes), formulated as an edge-labeling task.
register_task('edges-srl-conll2012', rel_path='edges/srl_conll2012',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Re-processed version of edges-coref-ontonotes, via AllenNLP data loaders.
register_task('edges-coref-ontonotes-conll',
               rel_path='edges/ontonotes-coref-conll',
               label_file="labels.txt", files_by_split={
                   'train': "coref_conll_ontonotes_en_train.json",
                   'val': "coref_conll_ontonotes_en_dev.json",
                   'test': "coref_conll_ontonotes_en_test.json",
               }, is_symmetric=False)(EdgeProbingTask)
# SPR1, as an edge-labeling task (multilabel).
register_task('edges-spr1', rel_path='edges/spr1',
               label_file="labels.txt", files_by_split={
                   'train': "spr1.train.json",
                   'val': "spr1.dev.json",
                   'test': "spr1.test.json",
               }, is_symmetric=False)(EdgeProbingTask)
# SPR2, as an edge-labeling task (multilabel).
register_task('edges-spr2', rel_path='edges/spr2',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Definite pronoun resolution. Two labels.
register_task('edges-dpr', rel_path='edges/dpr',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Relation classification on SemEval 2010 Task8. 19 labels.
register_task('edges-rel-semeval', rel_path='edges/semeval',
               label_file="labels.txt", files_by_split={
                   'train': "train.0.85.json",
                   'val': "dev.json",
                   'test': "test.json",
               }, is_symmetric=False)(EdgeProbingTask)

##
# New or experimental tasks.
##
# Relation classification on TACRED. 42 labels.
register_task('edges-rel-tacred', rel_path='edges/tacred/rel',
               label_file="labels.txt", files_by_split={
                   'train': "train.json",
                   'val': "dev.json",
                   'test': "test.json",
               }, is_symmetric=False)(EdgeProbingTask)

##
# Older tasks or versions for backwards compatibility.
##
# Entity classification on TACRED. 17 labels.
# NOTE: these are probably silver labels from CoreNLP,
# so this is of limited use as a target.
register_task('edges-ner-tacred', rel_path='edges/tacred/entity',
               label_file="labels.txt", files_by_split={
                   'train': "train.json",
                   'val': "dev.json",
                   'test': "test.json",
               }, single_sided=True)(EdgeProbingTask)
# SRL CoNLL 2005, formulated as an edge-labeling task.
register_task('edges-srl-conll2005', rel_path='edges/srl_conll2005',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.wsj.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Coreference on OntoNotes corpus. Two labels.
register_task('edges-coref-ontonotes', rel_path='edges/ontonotes-coref',
               label_file="labels.txt", files_by_split={
                   'train': "train.edges.json",
                   'val': "dev.edges.json",
                   'test': "test.edges.json",
               }, is_symmetric=False)(EdgeProbingTask)
# Entity type labeling on CoNLL 2003.
register_task('edges-ner-conll2003', rel_path='edges/ner_conll2003',
               label_file="labels.txt", files_by_split={
                   'train': "CoNLL-2003_train.json",
                   'val': "CoNLL-2003_dev.json",
                   'test': "CoNLL-2003_test.json",
               }, single_sided=True)(EdgeProbingTask)
# Dependency edge labeling on UD treebank (GUM). Use 'ewt' version instead.
register_task('edges-dep-labeling', rel_path='edges/dep',
               label_file="labels.txt", files_by_split={
                   'train': "train.json",
                   'val': "dev.json",
                   'test': "test.json",
               }, is_symmetric=False)(EdgeProbingTask)
# PTB constituency membership / labeling.
register_task('edges-constituent-ptb', rel_path='edges/ptb-membership',
               label_file="labels.txt", files_by_split={
                   'train': "ptb_train.json",
                   'val': "ptb_dev.json",
                   'test': "ptb_test.json",
               }, single_sided=True)(EdgeProbingTask)
# Constituency membership / labeling on OntoNotes.
register_task('edges-constituent-ontonotes',
               rel_path='edges/ontonotes-constituents',
               label_file="labels.txt", files_by_split={
                   'train': "consts_ontonotes_en_train.json",
                   'val': "consts_ontonotes_en_dev.json",
                   'test': "consts_ontonotes_en_test.json",
               }, single_sided=True)(EdgeProbingTask)
# CCG tagging (tokens only).
register_task('edges-ccg-tag', rel_path='edges/ccg_tag',
               label_file="labels.txt", files_by_split={
                   'train': "ccg.tag.train.json",
                   'val': "ccg.tag.dev.json",
                   'test': "ccg.tag.test.json",
               }, single_sided=True)(EdgeProbingTask)
# CCG parsing (constituent labeling).
register_task('edges-ccg-parse', rel_path='edges/ccg_parse',
               label_file="labels.txt", files_by_split={
                   'train': "ccg.parse.train.json",
                   'val': "ccg.parse.dev.json",
                   'test': "ccg.parse.test.json",
               }, single_sided=True)(EdgeProbingTask)

