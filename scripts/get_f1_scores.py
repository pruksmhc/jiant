import pandas as pd 
import torch
from ast import literal_eval
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse
import sys 

def get_accuracy_and_f1_scores(path_to_run, task_name):
	try:
		val = pd.read_csv(path_to_run+task_name+"_val.tsv", sep="\t")
		labels = val["true_label"].apply(lambda x: literal_eval(x))
		labels = torch.LongTensor(labels.tolist())
		labels = labels.squeeze(dim=1)
		labels = torch.argmax(labels, dim=1)
		preds = val["prediction"].tolist()
		if task_name == 'ultrafine-balanced':
			# now we only get finer statistics
			tagmask = val["category"].apply(lambda x: literal_eval(x)).tolist()
			tagmask = torch.LongTensor(tagmask)
			tagmask = torch.argmax(tagmask, dim=1)
			indices = (tagmask == 2).nonzero().squeeze(-1).tolist()
			preds = val["prediction"][indices]
			labels = labels[indices]
		print("Validation results")
		print("Accuracy:")
		print(accuracy_score(preds, labels))
		print('F1 score:')
		print(f1_score(preds, labels))
		print("confusion matrix")
		print("[TN, FP]")
		print("[FN, TP]")
		print(confusion_matrix(preds, labels))
	except Exception as e:
		pass
	val = pd.read_csv(path_to_run+task_name+"_test.tsv", sep="\t")
	labels = val["true_label"].apply(lambda x: literal_eval(x))
	labels = torch.LongTensor(labels.tolist())
	labels = labels.squeeze(dim=1)
	labels = torch.argmax(labels, dim=1)
	preds = val["prediction"].tolist()
	if task_name == 'ultrafine-balanced':
		# now we only get finer statistics
		tagmask = val["category"].apply(lambda x: literal_eval(x)).tolist()
		tagmask = torch.LongTensor(tagmask)
		tagmask = torch.argmax(tagmask, dim=1)
		indices = (tagmask == 2).nonzero().squeeze(-1).tolist()
		preds = val["prediction"][indices]
		labels = labels[indices]
	print("Test results")
	print("Accuracy:")
	print(accuracy_score(preds, labels))
	print('F1 score:')
	print(f1_score(preds, labels))
	print("confusion matrix")
	print("[TN, FP]")
	print("[FN, TP]")
	print(confusion_matrix(preds, labels))
	

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default=4,
                        help="Number of parallel processes to use.")
    parser.add_argument('--inputs', type=str, nargs="+",
                        help="Input JSON files.")
    args = parser.parse_args(args)
    get_accuracy_and_f1_scores(args.inputs[0], args.task_name)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)