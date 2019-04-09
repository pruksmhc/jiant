import torch
from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure, Average


my = F1Measure(positive_label =1)
logits = [[0, 1], [0,1], [0, 1], [1,0]]
labels = [1, 1, 0, 0]
logits = torch.LongTensor(logits)
labels = torch.LongTensor(labels)
binary_scores = torch.stack([-1 * logits, logits], dim=2)
my(logits, labels)
# abels_in_int_format = torch.max(labels, dim=1)[1] -you have to do this for microf1 btw
# to mae it correct 

class MacroF1():
  def __init__(self, beta=1.0):
    self.true_pos_count = 0
    self.true_neg_count = 0
    self.pred_pos_count = 0 
    self.pred_neg_count = 0
    self.correct_pos_predictions_count = 0 
    self.correct_neg_predictions_count = 0 
    self.beta = beta

  def __call__(self, logits, labels):
    logits_ints =  torch.max(logits, dim=1)[1]
    self.true_pos_count += len(labels.nonzero().squeeze(1))
    self.true_neg_count += len((labels == 0).nonzero().squeeze(1))
    self.pred_pos_count += len(logits_ints.nonzero().squeeze(1))
    self.pred_neg_count += len((logits_ints == 0).nonzero().squeeze(1))
    pos_indices = labels.nonzero().squeeze(1)
    pos_preds = logits_ints[pos_indices]
    self.correct_pos_predictions_count += len((pos_preds == labels[pos_indices]).nonzero())
    neg_indices = (labels== 0).nonzero().squeeze(1)
    neg_preds = logits_ints[neg_indices]
    self.correct_neg_predictions_count += len((neg_preds == labels[neg_indices]).nonzero())

  def get_metric(self, reset=False):
    import numpy as np
    pred_sum = np.array([self.pred_neg_count, self.pred_pos_count])
    true_sum = np.array([self.true_neg_count, self.true_pos_count])
    tp_sum = np.array([self.correct_neg_predictions_count, self.correct_pos_predictions_count ])
    precision = np.divide(tp_sum, pred_sum)
    recall = np.divide(tp_sum, true_sum)
    denom = self.beta * precision + recall
    denom[denom == 0.] = 1
    f_score = (1 + self.beta) * precision * recall / denom
    f_score = np.average(f_score)
    return torch.FloatTensor(f_score)
 
