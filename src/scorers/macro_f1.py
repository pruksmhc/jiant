import torch
from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure, Average
import logging as log

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
    pos_indices = labels.nonzero().squeeze(1).cuda()
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
    return torch.from_numpy(np.array(f_score))


"""
logits = torch.Tensor([[0.01, 0.234],[0.01, 0.234], [0.01, 0.234], [0.8, 0.2]])
labels = torch.Tensor([1, 1, 0, 0])
pred = torch.nn.Softmax(dim=1)(logits)
pred = torch.argmax(pred, dim=1)
def one_hot_v(batch, depth=2):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)
one_hot_logits = one_hot_v(pred)
micro_f1_scorer = F1Measure(positive_label=1)
macro_f1_scorer = MacroF1()
micro_f1_scorer(one_hot_v(pred).long(), labels.long())
macro_f1_scorer(one_hot_logits.long(), labels.long())
print("our version of micro f1")
print(micro_f1_scorer.get_metric())
print("our version of macro f1")
print(macro_f1_scorer.get_metric())
y_pred = [1, 1, 1, 0]
y_true = [1, 1, 0, 0]
import sklearn 
from sklearn.metrics import f1_score
print("sklearn version of micro f1")
print(f1_score(y_true, y_pred))
print("sklearn version of macro f1")
print(f1_score(y_true, y_pred, average='macro'))
"""
