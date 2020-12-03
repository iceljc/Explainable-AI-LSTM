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
import heapq, random

def deletion_iter(num_del, method, linear, alpha, data_iter, net):
	correct_total, incorrect_total = 0, 0
	count1, count2 = 0, 0
	backward = True
	print("backward: ", backward)

	for i, batch in enumerate(data_iter):
		src = batch.src.cpu().numpy()[0]
		true_label = batch.trg.cpu().numpy()[0]

		if len(src) >= 10:
			# print(correct_total, incorrect_total)
			net.set_input(src)
			probs = net.forward()
			pred_label = np.argmax(probs)

			if pred_label == true_label:
				correct_total += 1

				if backward:
					Gx, Gx_rev = net.backward(pred_label)
					R_words = (np.linalg.norm(Gx + Gx_rev, ord=2, axis=1))**2
				else:
					Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha)
					R_words = np.sum(Rx_fw + Rx_bw, axis=1)

				# delete in descending order
				scores = [(r, i) for i, r in enumerate(R_words)]
				scores.sort(key=lambda x: x[0], reverse=True)
				deletion = np.array([i for r, i in scores[:num_del]])

				net.set_input(src, delete_ids=deletion)
				probs_del = net.forward()
				pred_del = np.argmax(probs_del)

				if pred_del == true_label:
					count1 += 1


				# words = lookup_words(src, vocab)
			else:
				incorrect_total += 1

				if backward:
					Gx, Gx_rev = net.backward(pred_label)
					R_words = (np.linalg.norm(Gx + Gx_rev, ord=2, axis=1))**2
				else:
					Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha)
					R_words = np.sum(Rx_fw + Rx_bw, axis=1)

				# delete in descending order
				scores = []
				for i, r in enumerate(R_words):
					scores.append((r, i))
				scores.sort(key=lambda x: x[0], reverse=True)
				deletion = np.array([i for r, i in scores[:num_del]])

				net.set_input(src, delete_ids=deletion)
				probs_del = net.forward()
				pred_del = np.argmax(probs_del)

				if pred_del == true_label:
					count2 += 1

	print("Number of deletion: ", num_del)
	print(count1, correct_total)
	print(count2, incorrect_total)
	print("Accuracy (correct initially): {0:0.2f}".format(count1 / correct_total * 100))
	print("Accuracy (false initially): {0:0.2f}".format(count2 / incorrect_total * 100))
	print()



def run_lrp(data_iter, vocab, model_file):
	net = BiLSTM(model_file=model_file)
	method = 'all'
	linear = 'abs'
	if linear == 'zero':
		alpha = 2
	else:
		alpha = 1

	print("LRP - ", method)
	print("Linear - ", linear)
	print("alpha - ", alpha)
	num_del = 1

	while num_del <= 5:
		iterator = (rebatch(PAD_INDEX, b) for b in data_iter)
		deletion_iter(num_del, method, linear, alpha, iterator, net)
		num_del += 1
	

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

	print_data_info(train_data, valid_data, test_data, SRC, LABEL)

	#############################
	

	run_lrp(test_iter, vocab=SRC.vocab, model_file='sa_model4.pt')

	










