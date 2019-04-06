import torch
from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure, Average

class MacroF1():
  def __init__(self):
    self.f1_pos_class = F1Measure(positive_label=1)
    self.f1_neg_class = F1Measure(positive_label=1)

  def __call__(self, logits, labels):
    labels_in_int_format = torch.max(labels, dim=1)[1]
    pos_idx = torch.nonzero(labels_in_int_format).squeeze(dim=1) # find all positive class
    if len(pos_idx) > 0:
        self.f1_pos_class(logits[pos_idx], labels[pos_idx])
    neg_idx = torch.nonzero(labels_in_int_format == 0).squeeze(dim=1)
    if len(neg_idx) > 0:
        self.f1_neg_class(logits[neg_idx],labels[neg_idx])

  def get_metric(self, reset=False):
    macro_f1 = (self.f1_pos_class.get_metric(reset)[2] + self.f1_neg_class.get_metric(reset)[2]) / 2.0
    return macro_f1
