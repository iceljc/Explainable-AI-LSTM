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

def replace_iter(num_replace, method, data_iter, net, vocab_size):
	correct_total, incorrect_total = 0, 0
	count1, count2 = 0, 0
	print(num_replace)

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
				# Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
				# R_words = np.sum(Rx_fw + Rx_bw, axis=1)

				# random replacement
				replace_ids = random.sample(range(0, len(src)), num_replace)
				replace_words = random.sample(range(0, vocab_size), num_replace)
				
				for i in range(num_replace):
					idx = replace_ids[i]
					src[idx] = replace_words[i]

				net.set_input(src)
				probs_rand = net.forward()
				pred_rand = np.argmax(probs_rand)

				if pred_rand == true_label:
					count1 += 1


				# words = lookup_words(src, vocab)
			else:
				incorrect_total += 1
				# Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
				# R_words = np.sum(Rx_fw + Rx_bw, axis=1)

				# random replacement
				replace_ids = random.sample(range(0, len(src)), num_replace)
				replace_words = random.sample(range(0, vocab_size), num_replace)
				
				for i in range(num_replace):
					idx = replace_ids[i]
					src[idx] = replace_words[i]

				net.set_input(src)
				probs_rand = net.forward()
				pred_rand = np.argmax(probs_rand)

				if pred_rand == true_label:
					count2 += 1

	# print("Number of replacement: ", num_replace)
	# print(count1, correct_total)
	# print(count2, incorrect_total)
	# print("Accuracy (correct initially): {0:0.2f}".format(count1 / correct_total * 100))
	# print("Accuracy (false initially): {0:0.2f}".format(count2 / incorrect_total * 100))
	# print()

	return count1, count2, correct_total, incorrect_total



def run_lrp(data_iter, vocab, model_file):
	net = BiLSTM(model_file=model_file)
	vocab_size = len(vocab)
	method = None
	# print("LRP - ", method)
	num_replace = 1
	iters = 8
	result_true, result_false = {}, {}

	while num_replace <= 5:
		result_true[num_replace] = []
		result_false[num_replace] = []

		for it in range(iters):
			iterator = (rebatch(PAD_INDEX, b) for b in data_iter)
			count1, count2, correct_total, incorrect_total = replace_iter(num_replace, method, iterator, net, vocab_size)
			result_true[num_replace].append(count1)
			result_false[num_replace].append(count2)

		num_replace += 1

	print("Initially correct: ", result_true)
	print("Initially false: ", result_false)

	count_true, count_false = 0, 0
	for k in result_true:
		count_true += sum(result_true[k]) / len(result_true[k])
		count_false += sum(result_false[k]) / len(result_false[k])

		print("Number of replacement: ", k)
		print("Accuracy (initially correct): {0:0.2f}".format(count_true/correct_total*100))
		print("Accuracy (initially false): {0:0.2f}".format(count_false/incorrect_total*100))

		count_true, count_false = 0, 0
	

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

	










