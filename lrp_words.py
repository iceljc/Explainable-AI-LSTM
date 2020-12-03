import torch
import torch.nn as nn
import numpy as np
import math, copy, time
from torchtext import data, datasets
from setting import params
from utils import print_data_info, print_examples, count_correct_prediction, lookup_words
from model import make_model, SimpleLossCompute
from init_weight import init_weight
from data import rebatch
from biLSTM import *
import heapq


def run_lrp(data_iter, vocab, model_file):
	net = BiLSTM(model_file=model_file)
	method = 'all'
	linear = 'abs'
	backward = True
	print("backward: ", backward)
	if linear == 'zero':
		alpha = 2
	else:
		alpha = 1

	print("LRP - ", method)
	print("Linear: ", linear)
	print("alpha - ", alpha)
	min_h, max_h = [], []
	most, least = set(), set()
	size = 10
	target_label = 2
	print("Target label: ", target_label)

	for i, batch in enumerate(data_iter):
		src = batch.src.cpu().numpy()[0]
		true_label = batch.trg.cpu().numpy()[0]
		net.set_input(src)
		probs = net.forward()
		pred_label = np.argmax(probs)

		if pred_label == target_label:

			if backward:
				Gx, Gx_rev = net.backward(pred_label)
				R_words = (np.linalg.norm(Gx + Gx_rev, ord=2, axis=1))**2
			else:
				Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha, eps=0.001, bias_factor=0.0)
				R_words = np.sum(Rx_fw + Rx_bw, axis=1)

			words = lookup_words(src, vocab)

			for i,w in enumerate(words):
				if w not in most:
					heapq.heappush(min_h, (R_words[i], w))
					most.add(w)
				if w not in least:
					heapq.heappush(max_h, (-R_words[i], w))
					least.add(w)

				if len(min_h) > size:
					heapq.heappop(min_h)
					most = set([x for r,x in min_h])
				
				if len(max_h) > size:
					heapq.heappop(max_h)
					least = set([x for r,x in max_h])

	
	print("Most relevant words: ")
	while min_h:
		r, w = heapq.heappop(min_h)
		print(w, r)

	print()
	print("Least relevant words: ")
	while max_h:
		r, w = heapq.heappop(max_h)
		print(w, -r)






if __name__ == '__main__':
	
	seed = 42
	# np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	DEVICE = params['DEVICE']

	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"    
	# SOS_TOKEN = "<s>"
	# EOS_TOKEN = "</s>"
	LOWER = True

	SRC = data.Field(sequential=True, tokenize='spacy', batch_first=True, lower=LOWER, include_lengths=True, 
		unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)
	# , init_token=SOS_TOKEN, eos_token=EOS_TOKEN

	LABEL = data.Field(sequential=False)

	#############################
	train_data, valid_data, test_data = datasets.SST.splits(SRC, LABEL)

	# build vocab
	SRC.build_vocab(train_data, max_size=30000,vectors="glove.6B.300d", unk_init=torch.Tensor.normal_)
	LABEL.build_vocab(train_data)

	PAD_INDEX = SRC.vocab.stoi[PAD_TOKEN]
	# SOS_INDEX = SRC.vocab.stoi[SOS_TOKEN]
	# EOS_INDEX = SRC.vocab.stoi[EOS_TOKEN]
	# print(LABEL.vocab.freqs.most_common(10))

	#############################
	# define iterator
	train_iter = data.BucketIterator(train_data, batch_size=params['BATCH_SIZE'],
		device=DEVICE, sort_within_batch=True, 
		sort_key=lambda x: len(x.text), 
		train=True, repeat=False)

	# train_iter = data.Iterator(train_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	# print_data_info(train_data, valid_data, test_data, SRC, LABEL)

	#############################
	

	run_lrp((rebatch(PAD_INDEX, b) for b in test_iter), vocab=SRC.vocab, model_file='sa_model4.pt')

	










