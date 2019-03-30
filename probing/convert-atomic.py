"""
Converts  ATOMIC to 
sentence: [ ] 
Input 2: 
-Make it a binary choice. 
Binary choice, and it needs to choose it. Do a sigmoid. 
And then it needs to figure out which one makes sense. 
lable: 
"""
import pandas as pd
import random
import math
import itertools
from ast import literal_eval

def convert_atomic():
	result_text = []
	current_text = pd.read_csv("/beegfs/yp913/jiant/data/atomic/v4_atomic_all.csv")
	reaction_type_to_values = {col: list(set(current_text[col].values)) for col in current_text.columns}
	import pdb; pdb.set_trace()
	for i in range(len(current_text)):
		row = current_text.iloc[i]
		text = row["event"]
		for name in current_text.columns:
			if name.startswith("o") or name.startswith('x'):
				good_options = row[name]
				if good_options != '[]' :
					# randomly select another, make it balanced. 
					all_options = reaction_type_to_values[name]
					context = [row[s] for s in current_text.columns if ((s.startswith("o") or s.startswith("x")) and s!= name)]
					fin_context = [c for c in context if c != '[]']
					fin_context = [literal_eval(c) for c in fin_context]
					fin_context  = list(itertools.chain.from_iterable(fin_context))
					choice = random.randint(1,len(all_options) - 1)
					bad_option = all_options[choice]
					result_text.append([text, " ".join(literal_eval(good_options)), " ".join(fin_context), "true", name])
					result_text.append([text, " ".join(literal_eval(bad_option)), " ".join(fin_context), "false", name])
	import pdb; pdb.set_trace()
	random.shuffle(result_text)
	train_set = result_text[:math.ceil(len(result_text) * 0.8)]
	val_set = result_text[math.ceil(len(result_text) * 0.8): math.ceil(len(result_text) * 0.9)]
	test_set = result_text[math.ceil(len(result_text) * 0.9):]
	train = pd.DataFrame(train_set, columns=["text", "input2", "context", "label", "category"])
	train.to_csv("/beegfs/yp913/jiant/data/atomic/train.csv")
	val = pd.DataFrame(val_set, columns=["text", "input2", "context", "label", "category"])
	val.to_csv("/beegfs/yp913/jiant/data/atomic/val.csv")
	test = pd.DataFrame(test_set, columns=["text", "input2", "context", "label", "category"])
	test.to_csv("/beegfs/yp913/jiant/data/atomic/test.csv")

convert_atomic()
